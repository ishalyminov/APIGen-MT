#!/usr/bin/env python3
"""Cache frozen-base assistant-token log probabilities once for a GSPO sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from datasets import Sequence, Value, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-segments", type=int, default=256)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def validate_prepared_files(dataset_dir: Path, manifest: dict) -> None:
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


def main() -> int:
    args = parse_args()
    configure_logprob_runtime()
    if args.batch_size < 1 or args.checkpoint_segments < args.batch_size:
        raise ValueError(
            "batch size must be positive and checkpoint size must cover a batch"
        )
    source_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    validate_prepared_files(args.dataset, source_manifest)
    source_tree_sha = tree_sha256(args.dataset)
    expected_tree_sha = source_manifest.get("dataset_tree_sha256")
    if expected_tree_sha and source_tree_sha != expected_tree_sha:
        raise ValueError(
            f"prepared dataset changed: {source_tree_sha} != {expected_tree_sha}"
        )
    model_config_sha = file_sha256(args.model / "config.json")
    weight_hashes = model_weight_hashes(args.model)
    scorer_contract = logprob_runtime_contract(args.attn_implementation)
    scorer_contract["precompute_script_sha256"] = file_sha256(Path(__file__))
    manifest_path = args.output.with_suffix(".manifest.json")
    done_path = args.output.with_suffix(".done")
    if done_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("source_dataset_tree_sha256") != source_tree_sha
            or manifest.get("model_config_sha256") != model_config_sha
            or manifest.get("model_weight_sha256") != weight_hashes
            or manifest.get("scorer_contract") != scorer_contract
            or manifest.get("output_dataset_tree_sha256") != tree_sha256(args.output)
        ):
            raise ValueError("existing old-logprob cache contract mismatch")
        print(f"already complete: {args.output}")
        return 0
    if args.output.exists():
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            output_sha = tree_sha256(args.output)
            if (
                manifest.get("source_dataset_tree_sha256") == source_tree_sha
                and manifest.get("model_config_sha256") == model_config_sha
                and manifest.get("model_weight_sha256") == weight_hashes
                and manifest.get("scorer_contract") == scorer_contract
                and manifest.get("output_dataset_tree_sha256") == output_sha
            ):
                done_path.write_text(output_sha + "\n", encoding="utf-8")
                print(f"recovered completed cache marker: {args.output}")
                return 0
        raise FileExistsError(f"refusing unverified existing output: {args.output}")
    staged_output = args.output.with_name(args.output.name + ".building")
    if staged_output.exists():
        archived = staged_output.with_name(
            staged_output.name + f".abandoned.{int(time.time())}"
        )
        os.replace(staged_output, archived)
        print(f"archived interrupted final save: {archived}", flush=True)

    dataset = load_from_disk(str(args.dataset))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    print(f"loading frozen behavior policy: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).cuda()
    model.eval()

    # Similar-length batching sharply reduces padding while retaining output
    # row order.  Reading one length at a time avoids materializing ~174M token
    # IDs as Python integers just to sort 53k rows.
    lengths = [len(dataset[index]["input_ids"]) for index in range(len(dataset))]
    ordered_indices = sorted(range(len(dataset)), key=lengths.__getitem__)
    rows_with_logps: list[list[float] | None] = [None] * len(dataset)
    progress_dir = args.output.with_suffix(".progress")
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_identity = {
        "version": "offline_gspo_old_logps_progress_v1",
        "source_dataset_tree_sha256": source_tree_sha,
        "source_manifest_sha256": file_sha256(args.dataset_manifest),
        "model_config_sha256": model_config_sha,
        "model_weight_sha256": weight_hashes,
        "scorer_contract": scorer_contract,
        "rows": len(dataset),
        "ordered_indices_sha256": hashlib.sha256(
            json.dumps(ordered_indices, separators=(",", ":")).encode()
        ).hexdigest(),
        "checkpoint_segments": args.checkpoint_segments,
    }
    identity_path = progress_dir / "identity.json"
    if identity_path.exists():
        existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing_identity != progress_identity:
            raise ValueError("old-logprob progress belongs to a different contract")
    else:
        identity_path.write_text(
            json.dumps(progress_identity, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    started = time.time()
    with torch.inference_mode():
        for part_start in range(0, len(ordered_indices), args.checkpoint_segments):
            part_indices = ordered_indices[
                part_start : part_start + args.checkpoint_segments
            ]
            part_number = part_start // args.checkpoint_segments
            part_path = progress_dir / f"part_{part_number:05d}.pt"
            if part_path.exists():
                payload = torch.load(part_path, map_location="cpu", weights_only=False)
                if payload.get("indices") != part_indices:
                    raise ValueError(f"progress index mismatch in {part_path}")
                part_logps = payload.get("logps")
                if not isinstance(part_logps, list) or len(part_logps) != len(part_indices):
                    raise ValueError(f"progress logprob shape mismatch in {part_path}")
            else:
                part_logps: list[torch.Tensor] = []
                for batch_start in range(0, len(part_indices), args.batch_size):
                    indices = part_indices[batch_start : batch_start + args.batch_size]
                    rows = [dataset[index] for index in indices]
                    batch = move_batch_to_device(
                        pad_segments(
                            rows,
                            pad_token_id=tokenizer.pad_token_id,
                            require_old_logps=False,
                        ),
                        torch.device("cuda"),
                    )
                    logps = selected_token_logps(
                        model=model,
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        target_rows=batch["target_rows"],
                        target_positions=batch["target_positions"],
                        temperatures=batch["temperatures"],
                    ).cpu()
                    offset = 0
                    for row in rows:
                        width = len(row["target_positions"])
                        part_logps.append(logps[offset : offset + width].clone())
                        offset += width
                    if offset != len(logps):
                        raise AssertionError(
                            "selected logprob split did not consume the batch"
                        )
                temporary_part = part_path.with_suffix(".pt.tmp")
                torch.save(
                    {"indices": part_indices, "logps": part_logps},
                    temporary_part,
                )
                os.replace(temporary_part, part_path)
            for dataset_index, values in zip(part_indices, part_logps):
                if values.ndim != 1 or not torch.isfinite(values).all():
                    raise ValueError(f"invalid cached logprobs in {part_path}")
                rows_with_logps[dataset_index] = values.tolist()
            done = min(part_start + len(part_indices), len(ordered_indices))
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"old-logps {done}/{len(dataset)} segments "
                f"({done/elapsed:.2f}/s, checkpoint={part_path.name})",
                flush=True,
            )

    if any(values is None for values in rows_with_logps):
        raise AssertionError("old-logprob cache row assignment is incomplete")
    completed_logps = [values for values in rows_with_logps if values is not None]
    output = dataset.add_column(
        "old_logps",
        completed_logps,
        feature=Sequence(Value("float32")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save_to_disk(str(staged_output))
    output_tree_sha = tree_sha256(staged_output)
    manifest = {
        "version": "offline_gspo_old_logps_v1",
        "created_at_unix": time.time(),
        "source_dataset": str(args.dataset.resolve()),
        "source_dataset_tree_sha256": source_tree_sha,
        "source_manifest": str(args.dataset_manifest.resolve()),
        "source_manifest_sha256": file_sha256(args.dataset_manifest),
        "model": str(args.model.resolve()),
        "model_config_sha256": model_config_sha,
        "model_weight_sha256": weight_hashes,
        "scorer_contract": scorer_contract,
        "rows": len(output),
        "target_tokens": sum(len(values) for values in completed_logps),
        "dtype": "float32",
        "temperature_policy": "per_archived_sampling_condition",
        "nominal_logprob_distribution": "temperature-scaled full softmax",
        "behavior_sampling_truncation_not_reconstructed": {
            "top_k": 20,
            "top_p": 0.95,
            "reason": "sampling-time token logprobs were not archived",
        },
        "batch_order": "stable length-sorted; restored to source row order",
        "resumable_progress_dir": str(progress_dir.resolve()),
        "checkpoint_segments": args.checkpoint_segments,
        "output_dataset_tree_sha256": output_tree_sha,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(staged_output, args.output)
    done_path.write_text(output_tree_sha + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
