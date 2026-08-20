"""Shared model/log-probability helpers for offline GSPO."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path

import torch
import transformers
from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearPPOBase


LOGPROB_FLOAT32_MATMUL_PRECISION = "high"


def configure_logprob_runtime() -> None:
    """Apply math settings shared by the frozen reference and actor."""

    torch.set_float32_matmul_precision(LOGPROB_FLOAT32_MATMUL_PRECISION)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def logprob_runtime_contract(attn_implementation: str) -> dict[str, object]:
    """Identity of code/runtime choices that can change nominal logprobs."""

    path = Path(__file__).resolve()
    contract: dict[str, object] = {
        "modeling_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "attn_implementation": attn_implementation,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "liger_kernel": _package_version("liger-kernel"),
        "causal_conv1d_available": importlib.util.find_spec("causal_conv1d")
        is not None,
        "fla_available": importlib.util.find_spec("fla") is not None,
    }
    if torch.cuda.is_available():
        contract.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(),
                "cuda_device_capability": list(
                    torch.cuda.get_device_capability()
                ),
            }
        )
    return contract


def pad_segments(
    rows: Sequence[dict],
    *,
    pad_token_id: int,
    require_old_logps: bool,
) -> dict[str, torch.Tensor | list[str]]:
    """Right-pad rendered user-turn segments and retain sparse target indices."""

    if not rows:
        raise ValueError("cannot collate an empty segment batch")
    max_length = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), max_length), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
    target_rows: list[int] = []
    target_positions: list[int] = []
    old_logps: list[float] = []
    for row_index, row in enumerate(rows):
        ids = [int(value) for value in row["input_ids"]]
        positions = [int(value) for value in row["target_positions"]]
        if not ids or not positions:
            raise ValueError("every segment needs input and assistant target tokens")
        if positions != sorted(set(positions)):
            raise ValueError("target_positions must be sorted and unique")
        if positions[0] <= 0 or positions[-1] >= len(ids):
            raise ValueError("target position cannot be BOS or past the segment")
        input_ids[row_index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row_index, : len(ids)] = 1
        target_rows.extend([row_index] * len(positions))
        target_positions.extend(positions)
        if require_old_logps:
            values = [float(value) for value in row["old_logps"]]
            if len(values) != len(positions):
                raise ValueError("old_logps must align with target_positions")
            old_logps.extend(values)
    result: dict[str, torch.Tensor | list[str]] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_rows": torch.tensor(target_rows, dtype=torch.long),
        "target_positions": torch.tensor(target_positions, dtype=torch.long),
        "temperatures": torch.tensor(
            [float(row["temperature"]) for row in rows], dtype=torch.float32
        ),
        "episode_ids": [str(row["episode_id"]) for row in rows],
    }
    if require_old_logps:
        old_tensor = torch.tensor(old_logps, dtype=torch.float32)
        if not torch.isfinite(old_tensor).all():
            raise ValueError("old_logps contains a non-finite value")
        result["old_logps"] = old_tensor
    return result


def selected_token_logps(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_rows: torch.Tensor,
    target_positions: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    """Compute archived assistant-token logps without full vocab logits.

    Qwen3.5's hybrid linear-attention implementation is not numerically padding
    invariant: scoring the same sequence beside a longer right-padded row moved
    log probabilities by ~2e-2, far larger than GSPO's 3e-4 clip.  Therefore
    every rendered segment is forwarded independently.  Multiple segments are
    still connected to one episode-level ratio by ``token_episode_ids`` in the
    loss; this only fixes the actor-collation contract.
    """

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must be equal 2D tensors")
    if temperatures.shape != (input_ids.shape[0],):
        raise ValueError("one temperature is required per rendered segment")
    if not torch.isfinite(temperatures).all() or torch.any(temperatures <= 0):
        raise ValueError("temperatures must be finite and positive")
    if target_rows.shape != target_positions.shape or target_rows.numel() == 0:
        raise ValueError("target row/position vectors must be non-empty and aligned")
    if int(target_rows.min()) < 0 or int(target_rows.max()) >= input_ids.shape[0]:
        raise ValueError("target row is out of range")
    logps = torch.empty(
        target_positions.shape[0], device=input_ids.device, dtype=torch.float32
    )
    for row_index in range(input_ids.shape[0]):
        token_indices = torch.nonzero(
            target_rows == row_index,
            as_tuple=False,
        ).flatten()
        if token_indices.numel() == 0:
            continue
        row_mask = attention_mask[row_index]
        active_length = int(row_mask.sum().item())
        if active_length <= 0 or not torch.all(row_mask[:active_length] == 1):
            raise ValueError("attention mask is not a non-empty right-padded prefix")
        if active_length < row_mask.numel() and not torch.all(
            row_mask[active_length:] == 0
        ):
            raise ValueError("attention mask is not right padded")
        row_positions = target_positions[token_indices]
        if int(row_positions.max()) >= active_length:
            raise ValueError("target position points into padding")
        row_input_ids = input_ids[row_index : row_index + 1, :active_length]
        row_attention_mask = attention_mask[
            row_index : row_index + 1, :active_length
        ]
        outputs = model.model(
            input_ids=row_input_ids,
            attention_mask=row_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        selected_hidden = hidden[0, row_positions - 1]
        selected_ids = row_input_ids[0, row_positions]
        values = LigerFusedLinearPPOBase.chunk_forward(
            selected_hidden.unsqueeze(0),
            model.lm_head.weight,
            selected_ids.unsqueeze(0),
            bias=getattr(model.lm_head, "bias", None),
            temperature=float(temperatures[row_index]),
        ).squeeze(0)
        logps = logps.index_copy(0, token_indices, values.float())
    if not torch.isfinite(logps).all():
        raise FloatingPointError("actor produced non-finite selected-token logps")
    return logps


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
