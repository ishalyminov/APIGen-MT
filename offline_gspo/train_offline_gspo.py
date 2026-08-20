#!/usr/bin/env python3
"""One-epoch canonical-replay GSPO training for Qwen3.5-2B."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from offline_gspo.loss import offline_gspo_loss
from offline_gspo.modeling import (
    configure_logprob_runtime,
    logprob_runtime_contract,
    move_batch_to_device,
    pad_segments,
    selected_token_logps,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def model_weight_hashes(path: Path) -> dict[str, str]:
    files = sorted(path.glob("*.safetensors"))
    index = path / "model.safetensors.index.json"
    if index.is_file():
        files.append(index)
    if not files:
        raise FileNotFoundError(f"model has no safetensors weights: {path}")
    return {item.name: file_sha256(item) for item in files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--base-model-contract", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--episodes-per-batch", type=int, default=1)
    parser.add_argument("--epsilon-low", type=float, default=3e-4)
    parser.add_argument("--epsilon-high", type=float, default=4e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--checkpoint-groups", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def build_episode_index(dataset) -> tuple[list[str], dict[str, list[int]], dict[str, dict[str, Any]]]:
    indices: dict[str, list[int]] = defaultdict(list)
    metadata: dict[str, dict[str, Any]] = {}
    for index in range(len(dataset)):
        row = dataset[index]
        episode_id = str(row["episode_id"])
        indices[episode_id].append(index)
        current = {
            "advantage": float(row["advantage"]),
            "reward": float(row["reward"]),
            "temperature": float(row["temperature"]),
            "group_id": str(row["group_id"]),
            "row_position": int(row["row_position"]),
            "sample_index": int(row["sample_index"]),
            "episode_segment_count": int(row["episode_segment_count"]),
            "episode_target_tokens": int(row["episode_target_tokens"]),
        }
        if episode_id in metadata and metadata[episode_id] != current:
            raise ValueError(f"inconsistent segment metadata for {episode_id}")
        metadata[episode_id] = current
    for episode_id, row_indices in indices.items():
        rows = [dataset[index] for index in row_indices]
        rows.sort(key=lambda row: int(row["segment_index"]))
        expected_count = metadata[episode_id]["episode_segment_count"]
        if len(rows) != expected_count:
            raise ValueError(
                f"episode {episode_id} has {len(rows)} segments, expected {expected_count}"
            )
        if [int(row["segment_index"]) for row in rows] != list(range(expected_count)):
            raise ValueError(f"episode {episode_id} segment indices are not contiguous")
        actual_tokens = sum(len(row["target_positions"]) for row in rows)
        if actual_tokens != metadata[episode_id]["episode_target_tokens"]:
            raise ValueError(f"episode {episode_id} target-token count changed")
        indices[episode_id] = sorted(
            row_indices, key=lambda index: int(dataset[index]["segment_index"])
        )
    return sorted(indices), dict(indices), metadata


def make_batch(
    *,
    dataset,
    episode_ids: list[str],
    episode_indices: dict[str, list[int]],
    episode_metadata: dict[str, dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
    require_old_logps: bool = False,
) -> tuple[dict, torch.Tensor]:
    rows = [
        dataset[index]
        for episode_id in episode_ids
        for index in episode_indices[episode_id]
    ]
    batch = move_batch_to_device(
        pad_segments(
            rows,
            pad_token_id=pad_token_id,
            require_old_logps=require_old_logps,
        ),
        device,
    )
    episode_to_local = {
        episode_id: index for index, episode_id in enumerate(episode_ids)
    }
    segment_episode_ids = torch.tensor(
        [episode_to_local[str(row["episode_id"])] for row in rows],
        dtype=torch.long,
        device=device,
    )
    token_episode_ids = segment_episode_ids[batch["target_rows"]]
    advantages = torch.tensor(
        [episode_metadata[episode_id]["advantage"] for episode_id in episode_ids],
        dtype=torch.float32,
        device=device,
    )
    batch["token_episode_ids"] = token_episode_ids
    return batch, advantages


class FP32MasterOptimizer:
    """AdamW over persistent FP32 masters for a BF16 actor.

    Keeping only Adam moments in FP32 is insufficient: the parameter update
    itself would still be rounded when applied directly to a BF16 parameter.
    Persistent masters accumulate sub-BF16 updates across optimizer steps and
    are copied to the actor after every step, which is standard mixed-precision
    training behaviour.
    """

    def __init__(self, model, *, lr: float, weight_decay: float) -> None:
        decay, no_decay = [], []
        self.pairs: list[tuple[torch.nn.Parameter, torch.nn.Parameter]] = []
        for _, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            master = torch.nn.Parameter(parameter.detach().float(), requires_grad=True)
            self.pairs.append((parameter, master))
            (decay if parameter.ndim >= 2 else no_decay).append(master)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=bool(self.pairs and self.pairs[0][1].is_cuda),
        )

    @property
    def parameters(self) -> list[torch.nn.Parameter]:
        return [master for _, master in self.pairs]

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def accumulate_model_grads(self, *, scale: float = 1.0) -> None:
        """Accumulate one BF16 microbatch gradient into FP32 group grads."""

        for actor, master in self.pairs:
            if actor.grad is None:
                continue
            update = actor.grad.detach().float().mul_(scale)
            if master.grad is None:
                master.grad = update
            else:
                master.grad.add_(update)

    @torch.no_grad()
    def sync_actor(self) -> None:
        for actor, master in self.pairs:
            actor.copy_(master)


def optimizer_for(model, *, lr: float, weight_decay: float) -> FP32MasterOptimizer:
    return FP32MasterOptimizer(model, lr=lr, weight_decay=weight_decay)


def actor_logps(model, batch: dict[str, Any]) -> torch.Tensor:
    """Compute policy log probabilities with the actor's BF16 weights."""

    return selected_token_logps(
        model=model,
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        target_rows=batch["target_rows"],
        target_positions=batch["target_positions"],
        temperatures=batch["temperatures"],
    )


def validate_prepared_files(dataset_dir: Path, manifest: dict[str, Any]) -> None:
    """Verify every immutable prepared-replay file against its signed hash."""

    expected = manifest.get("output_hashes_excluding_manifest")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("prepared manifest has no signed output-file hashes")
    actual_paths = {
        str(path.relative_to(dataset_dir))
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_paths != set(expected):
        raise ValueError("prepared dataset file set differs from its manifest")
    for relative, expected_sha in expected.items():
        actual_sha = file_sha256(dataset_dir / relative)
        if actual_sha != expected_sha:
            raise ValueError(
                f"prepared dataset file changed: {relative}; "
                f"expected={expected_sha} actual={actual_sha}"
            )


def save_resume_checkpoint(
    *,
    path: Path,
    master_optimizer: FP32MasterOptimizer,
    scheduler,
    next_group_offset: int,
    optimizer_step: int,
    global_step: int,
    contract: dict[str, Any],
) -> None:
    """Atomically save one optimizer-boundary checkpoint."""

    payload = {
        "version": "offline_gspo_train_resume_v1",
        "contract": contract,
        "next_group_offset": next_group_offset,
        "optimizer_step": optimizer_step,
        "global_step": global_step,
        "master_weights": [
            master.detach().cpu().clone()
            for _, master in master_optimizer.pairs
        ],
        "optimizer": master_optimizer.optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_resume_checkpoint(
    *,
    path: Path,
    master_optimizer: FP32MasterOptimizer,
    scheduler,
    expected_contract: dict[str, Any],
) -> tuple[int, int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != "offline_gspo_train_resume_v1":
        raise ValueError(f"unknown resume format: {path}")
    if payload.get("contract") != expected_contract:
        raise ValueError("training resume checkpoint contract mismatch")
    weights = payload.get("master_weights")
    if not isinstance(weights, list) or len(weights) != len(master_optimizer.pairs):
        raise ValueError("training resume master-weight shape mismatch")
    with torch.no_grad():
        for (_, master), saved in zip(master_optimizer.pairs, weights):
            if saved.shape != master.shape or saved.dtype != torch.float32:
                raise ValueError("training resume master tensor mismatch")
            master.copy_(saved)
    master_optimizer.optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    master_optimizer.sync_actor()
    torch.set_rng_state(payload["torch_rng_state"])
    if payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(payload["cuda_rng_state"])
    return (
        int(payload["next_group_offset"]),
        int(payload["optimizer_step"]),
        int(payload["global_step"]),
    )


def rollback_metric_log(path: Path, *, optimizer_step: int) -> None:
    """Discard diagnostics written after the last durable optimizer checkpoint."""

    if not path.exists():
        return
    retained: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            record = json.loads(line)
            record_step = int(record["optimizer_step"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid training metric record at {path}:{line_number}"
            ) from error
        if record_step <= optimizer_step:
            retained.append(line)
    temporary = path.with_suffix(path.suffix + ".rollback.tmp")
    temporary.write_text(
        "".join(f"{line}\n" for line in retained), encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.epochs != 1:
        raise ValueError("this conservative offline GSPO sweep requires one epoch")
    if args.learning_rate <= 0 or args.episodes_per_batch < 1:
        raise ValueError("learning rate and batch size must be positive")
    if args.checkpoint_groups < 1:
        raise ValueError("checkpoint interval must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output path: {args.output_dir}")
    staged_output = args.output_dir.with_name(args.output_dir.name + ".building")
    if staged_output.exists():
        archived = staged_output.with_name(
            staged_output.name + f".abandoned.{int(time.time())}"
        )
        os.replace(staged_output, archived)
        print(f"archived interrupted final save: {archived}", flush=True)

    configure_logprob_runtime()
    prepared_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    base_model_contract = json.loads(
        args.base_model_contract.read_text(encoding="utf-8")
    )
    if base_model_contract.get("version") != "offline_gspo_base_model_contract_v1":
        raise ValueError("unknown base-model contract version")
    prepared_manifest_sha = file_sha256(args.dataset_manifest)
    validate_prepared_files(args.dataset, prepared_manifest)
    source_hashes = (prepared_manifest.get("sources") or {}).get("hashes") or {}
    if file_sha256(args.system_prompt) != source_hashes.get("system_prompt"):
        raise ValueError("system prompt differs from prepared replay contract")
    if file_sha256(args.chat_template) != source_hashes.get("chat_template"):
        raise ValueError("chat template differs from prepared replay contract")
    dataset_tree_sha = tree_sha256(args.dataset)
    model_config_sha = file_sha256(args.model / "config.json")
    if model_config_sha != source_hashes.get("model_config"):
        raise ValueError("base model config differs from prepared replay tokenizer model")
    if model_config_sha != base_model_contract.get("config_sha256"):
        raise ValueError("base model config differs from pinned base-model contract")
    expected_tokenizer_hashes = base_model_contract.get("tokenizer_sha256")
    if not isinstance(expected_tokenizer_hashes, dict) or not expected_tokenizer_hashes:
        raise ValueError("base-model contract has no tokenizer hashes")
    for filename, expected_sha in expected_tokenizer_hashes.items():
        actual_sha = file_sha256(args.model / filename)
        prepared_sha = source_hashes.get(f"model/{filename}")
        if actual_sha != expected_sha or actual_sha != prepared_sha:
            raise ValueError(
                f"base tokenizer file differs from replay/contract: {filename}"
            )
    model_weights = model_weight_hashes(args.model)
    if model_weights != base_model_contract.get("model_weight_sha256"):
        raise ValueError("base weights differ from pinned rollout-policy contract")
    scorer_contract = logprob_runtime_contract(args.attn_implementation)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = load_from_disk(str(args.dataset))
    required_columns = {
        "input_ids",
        "target_positions",
        "episode_id",
        "group_id",
        "advantage",
        "reward",
        "temperature",
        "segment_index",
        "episode_segment_count",
        "episode_target_tokens",
        "row_position",
        "sample_index",
    }
    missing = required_columns - set(dataset.column_names)
    if missing:
        raise ValueError(f"offline GSPO dataset misses columns: {sorted(missing)}")
    episode_ids, episode_indices, episode_metadata = build_episode_index(dataset)
    prepared_counts = prepared_manifest.get("counts") or {}
    expected_counts = {
        "segments": len(dataset),
        "episodes": len(episode_ids),
        "groups": len({value["group_id"] for value in episode_metadata.values()}),
        "target_tokens": sum(
            value["episode_target_tokens"] for value in episode_metadata.values()
        ),
    }
    for name, actual in expected_counts.items():
        if int(prepared_counts.get(name, -1)) != actual:
            raise ValueError(
                f"prepared manifest {name} count changed: "
                f"manifest={prepared_counts.get(name)} actual={actual}"
            )
    print(
        json.dumps(
            {
                "segments": len(dataset),
                "episodes": len(episode_ids),
                "groups": len({value["group_id"] for value in episode_metadata.values()}),
                "positive_episodes": sum(value["reward"] > 0 for value in episode_metadata.values()),
                "dataset_tree_sha256": dataset_tree_sha,
            },
            indent=2,
        ),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    tokenizer.chat_template = args.chat_template.read_text(encoding="utf-8")
    device = torch.device("cuda")
    # Recompute the denominator from an immutable base policy in the same
    # process and on the same GPU as the actor. Qwen3.5 hybrid-attention logps
    # drifted by more than GSPO's 3e-4 clip across GPU/kernel runtimes, so an
    # externally cached denominator is not a valid proximal reference here.
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(device)
    reference_model.config.use_cache = False
    reference_model.requires_grad_(False)
    reference_model.eval()

    # The actor stays byte-compatible with the BF16 behavior policy.  AdamW is
    # backed by persistent FP32 master weights below so tiny sweep updates are
    # not rounded away.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()

    # Fail before spending an epoch unless two independently loaded base-policy
    # copies agree. Tiny GSPO clipping makes this gate materially important.
    preflight_ids = episode_ids[:1]
    preflight, preflight_advantages = make_batch(
        dataset=dataset,
        episode_ids=preflight_ids,
        episode_indices=episode_indices,
        episode_metadata=episode_metadata,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
    )
    with torch.no_grad():
        preflight_old = actor_logps(reference_model, preflight)
    preflight_new = actor_logps(model, preflight)
    token_diff = (preflight_new - preflight_old).detach()
    episode_sums = torch.zeros(len(preflight_ids), device=device)
    episode_counts = torch.zeros(len(preflight_ids), device=device)
    episode_sums.scatter_add_(0, preflight["token_episode_ids"], token_diff)
    episode_counts.scatter_add_(
        0, preflight["token_episode_ids"], torch.ones_like(token_diff)
    )
    episode_mean_diff = episode_sums / episode_counts
    token_tolerance = min(args.epsilon_low, args.epsilon_high) / 10
    if float(token_diff.abs().max()) > token_tolerance:
        raise ValueError(
            "frozen-reference/current token log-ratio mismatch is too large "
            f"for GSPO: max_abs={float(token_diff.abs().max())}"
        )
    if float(episode_mean_diff.abs().max()) > token_tolerance:
        raise ValueError(
            "base/current sequence log-ratio mismatch is too large for GSPO: "
            f"{episode_mean_diff.tolist()}"
        )
    preflight_loss, _ = offline_gspo_loss(
        new_logps=preflight_new,
        old_logps=preflight_old,
        token_episode_ids=preflight["token_episode_ids"],
        episode_advantages=preflight_advantages,
        epsilon_low=args.epsilon_low,
        epsilon_high=args.epsilon_high,
    )
    preflight_loss.backward()
    preflight_grad_norm = torch.sqrt(
        sum(
            parameter.grad.detach().float().square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    if not torch.isfinite(preflight_grad_norm) or float(preflight_grad_norm) == 0.0:
        raise ValueError(f"GSPO preflight has invalid grad norm: {preflight_grad_norm}")
    model.zero_grad(set_to_none=True)
    print(
        f"preflight ok: token_abs_max={float(token_diff.abs().max()):.6g} "
        f"episode_mean_abs_max={float(episode_mean_diff.abs().max()):.6g} "
        f"grad_norm={float(preflight_grad_norm):.6g}",
        flush=True,
    )
    del preflight, preflight_new, preflight_old, token_diff
    torch.cuda.empty_cache()

    master_optimizer = optimizer_for(
        model, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    optimizer = master_optimizer.optimizer
    group_episode_ids: dict[str, list[str]] = defaultdict(list)
    for episode_id in episode_ids:
        group_episode_ids[episode_metadata[episode_id]["group_id"]].append(episode_id)
    for group_id, members in group_episode_ids.items():
        if len(members) not in (31, 32):
            raise ValueError(
                f"GSPO group {group_id} has unexpected size {len(members)}"
            )
        temperatures = {episode_metadata[item]["temperature"] for item in members}
        if len(temperatures) != 1:
            raise ValueError(f"GSPO group {group_id} mixes temperatures")
        rewards = {episode_metadata[item]["reward"] for item in members}
        if not rewards.issuperset({0.0, 1.0}):
            raise ValueError(f"GSPO group {group_id} has no reward variance")
        advantages = [episode_metadata[item]["advantage"] for item in members]
        if abs(sum(advantages) / len(advantages)) > 2e-6:
            raise ValueError(f"GSPO group {group_id} advantages are not centered")
        sample_variance = sum(value * value for value in advantages) / (
            len(advantages) - 1
        )
        if abs(sample_variance - 1.0) > 2e-5:
            raise ValueError(
                f"GSPO group {group_id} advantages do not have sample std 1"
            )
    group_ids = sorted(group_episode_ids)
    optimizer_steps = len(group_ids) * args.epochs
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=round(optimizer_steps * args.warmup_ratio),
        num_training_steps=optimizer_steps,
    )
    order = list(group_ids)
    random.Random(args.seed).shuffle(order)
    resume_contract = {
        "dataset_tree_sha256": dataset_tree_sha,
        "prepared_manifest_sha256": prepared_manifest_sha,
        "base_model_contract_sha256": file_sha256(args.base_model_contract),
        "reference_policy": "online_frozen_base_same_process",
        "model_config_sha256": model_config_sha,
        "model_weight_sha256": model_weights,
        "learning_rate": args.learning_rate,
        "epsilon_low": args.epsilon_low,
        "epsilon_high": args.epsilon_high,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "episodes_per_batch": args.episodes_per_batch,
        "seed": args.seed,
        "group_order_sha256": hashlib.sha256(
            json.dumps(order, separators=(",", ":")).encode()
        ).hexdigest(),
        "logprob_scorer_contract": scorer_contract,
        "trainer_sha256": file_sha256(Path(__file__)),
        "loss_sha256": file_sha256(Path(__file__).with_name("loss.py")),
    }
    resume_path = args.output_dir.parent / f"{args.output_dir.name}.resume.pt"
    log_path = args.output_dir.parent / f"{args.output_dir.name}.train_metrics.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if resume_path.exists():
        start_group_offset, optimizer_step, global_step = load_resume_checkpoint(
            path=resume_path,
            master_optimizer=master_optimizer,
            scheduler=scheduler,
            expected_contract=resume_contract,
        )
        if not 0 <= start_group_offset <= len(order):
            raise ValueError("resume group offset is out of range")
        print(
            f"resuming at group {start_group_offset}/{len(order)} "
            f"from {resume_path}",
            flush=True,
        )
        rollback_metric_log(log_path, optimizer_step=optimizer_step)
        log_mode = "a"
    else:
        start_group_offset = 0
        global_step = 0
        optimizer_step = 0
        log_mode = "w"
    with log_path.open(log_mode, encoding="utf-8") as log_file:
        for group_offset in range(start_group_offset, len(order)):
                group_id = order[group_offset]
                members = group_episode_ids[group_id]
                group_size = len(members)
                metric_sums = defaultdict(float)
                model.zero_grad(set_to_none=True)
                master_optimizer.zero_grad()
                for episode_start in range(0, group_size, args.episodes_per_batch):
                    selected_episode_ids = members[
                        episode_start : episode_start + args.episodes_per_batch
                    ]
                    batch, advantages = make_batch(
                        dataset=dataset,
                        episode_ids=selected_episode_ids,
                        episode_indices=episode_indices,
                        episode_metadata=episode_metadata,
                        pad_token_id=tokenizer.pad_token_id,
                        device=device,
                    )
                    with torch.no_grad():
                        old_logps = actor_logps(reference_model, batch)
                    new_logps = actor_logps(model, batch)
                    loss, metrics = offline_gspo_loss(
                        new_logps=new_logps,
                        old_logps=old_logps,
                        token_episode_ids=batch["token_episode_ids"],
                        episode_advantages=advantages,
                        epsilon_low=args.epsilon_low,
                        epsilon_high=args.epsilon_high,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"non-finite GSPO loss in {group_id}: "
                            f"{float(loss.detach())}"
                        )
                    loss.backward()
                    microbatch_count = len(selected_episode_ids)
                    master_optimizer.accumulate_model_grads(
                        scale=microbatch_count / group_size
                    )
                    model.zero_grad(set_to_none=True)
                    metric_sums["loss"] += float(loss.detach()) * microbatch_count
                    metric_sums["sequence_ratio"] += (
                        float(metrics.mean_sequence_ratio) * microbatch_count
                    )
                    metric_sums["clip_fraction"] += (
                        float(metrics.clip_fraction) * microbatch_count
                    )
                    metric_sums["sampled_token_kl"] += (
                        float(metrics.sampled_token_kl) * microbatch_count
                    )
                    global_step += 1
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    master_optimizer.parameters,
                    args.max_grad_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                master_optimizer.sync_actor()
                scheduler.step()
                optimizer_step += 1
                if optimizer_step == 1 or optimizer_step % args.logging_steps == 0:
                    record = {
                        "epoch": 0,
                        "batch_step": global_step,
                        "optimizer_step": optimizer_step,
                        "optimizer_steps": optimizer_steps,
                        "group_id": group_id,
                        "group_episodes": group_size,
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": float(grad_norm),
                        "elapsed_seconds": time.time() - started,
                        **{
                            key: value / group_size
                            for key, value in metric_sums.items()
                        },
                    }
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()
                    print(json.dumps(record), flush=True)
                next_group_offset = group_offset + 1
                if (
                    next_group_offset % args.checkpoint_groups == 0
                    or next_group_offset == len(order)
                ):
                    save_resume_checkpoint(
                        path=resume_path,
                        master_optimizer=master_optimizer,
                        scheduler=scheduler,
                        next_group_offset=next_group_offset,
                        optimizer_step=optimizer_step,
                        global_step=global_step,
                        contract=resume_contract,
                    )
                    print(
                        f"saved resume checkpoint: {next_group_offset}/{len(order)}",
                        flush=True,
                    )

    staged_output.mkdir(parents=True, exist_ok=False)
    model.gradient_checkpointing_disable()
    model.zero_grad(set_to_none=True)
    del optimizer, scheduler, master_optimizer, reference_model
    torch.cuda.empty_cache()
    model.save_pretrained(
        staged_output,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(staged_output)
    template_text = args.chat_template.read_text(encoding="utf-8")
    (staged_output / "chat_template.jinja").write_text(
        template_text, encoding="utf-8"
    )
    contract = {
        "version": "offline_gspo_canonical_replay_v2",
        "objective": "gspo_sequence_ratio",
        "reference": "arxiv:2507.18071 equations 5-7",
        "base_model": str(args.model.resolve()),
        "base_model_config_sha256": model_config_sha,
        "base_model_weight_sha256": model_weights,
        "base_model_contract": str(args.base_model_contract.resolve()),
        "base_model_contract_sha256": file_sha256(args.base_model_contract),
        "logprob_scorer_contract": scorer_contract,
        "trainer_sha256": file_sha256(Path(__file__)),
        "loss_sha256": file_sha256(Path(__file__).with_name("loss.py")),
        "dataset": str(args.dataset.resolve()),
        "dataset_tree_sha256": dataset_tree_sha,
        "dataset_manifest": str(args.dataset_manifest.resolve()),
        "dataset_manifest_sha256": file_sha256(args.dataset_manifest),
        "reference_policy_mode": "online_frozen_base_same_process_and_gpu",
        "system_prompt_sha256": hashlib.sha256(
            args.system_prompt.read_text(encoding="utf-8").strip().encode()
        ).hexdigest(),
        "chat_template_sha256": hashlib.sha256(template_text.encode()).hexdigest(),
        "enable_thinking": True,
        "episodes": len(episode_ids),
        "segments": len(dataset),
        "groups": len({value["group_id"] for value in episode_metadata.values()}),
        "optimizer_unit": "one_complete_homogeneous_pass32_group",
        "resumable_checkpoint_groups": args.checkpoint_groups,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "epsilon_low": args.epsilon_low,
        "epsilon_high": args.epsilon_high,
        "kl_loss": False,
        "temperature_policy": "per_homogeneous_pass32_group",
        "canonical_replay_caveat": (
            "Parsed native calls were canonically re-rendered because original "
            "successful XML bytes and behavior logprobs were not archived. "
            "Ratios use temperature-scaled full-softmax nominal probabilities; "
            "the rollout sampler's top_k=20/top_p=0.95 truncation cannot be "
            "reconstructed. This is canonical nominal-policy offline GSPO, not "
            "exact behavior-policy importance sampling."
        ),
    }
    (staged_output / "toolonly_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (staged_output / ".train_done").write_text("ok\n", encoding="utf-8")
    os.replace(staged_output, args.output_dir)
    # The final model + immutable contract supersede the very large transient
    # optimizer checkpoint (roughly 24 GB for 2B); do not retain it forever.
    resume_path.unlink(missing_ok=True)
    resume_path.with_suffix(resume_path.suffix + ".tmp").unlink(missing_ok=True)
    print(json.dumps(contract, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
