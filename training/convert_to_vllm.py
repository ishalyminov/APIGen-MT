#!/usr/bin/env python3
"""Convert a text-only Qwen3.5 SFT checkpoint into a vLLM-servable VL checkpoint.

Why this exists
---------------
``train_toolcalling.py`` loads the base model via ``AutoModelForCausalLM`` and
saves a TEXT-only checkpoint: ``config.json`` declares ``Qwen3_5ForCausalLM``
(expecting ``model.layers.*``) while the Liger-patched weights are stored under
``model.language_model.*``. vLLM cannot serve that:
  * the text-only path mismatches keys ("Following weights were not initialized
    from checkpoint"), and
  * the Qwen3.5 text path hits ``NotImplementedError: page size not divisible``
    because of the hybrid (linear_attention + full_attention) KV-cache paging.

The working workaround (already used by your ``_vllm`` / ``_vlserving`` dirs) is
to serve via the VL architecture ``Qwen3_5ForConditionalGeneration``: the VL path
expects ``model.language_model.*`` and does not suffer the paging bug. VL models
serve plain text fine; the visual tower is simply never exercised.

This script rebuilds that serving dir by:
  1. taking ALL weights from the base VL model (language_model + visual + mtp),
  2. overwriting the ``model.language_model.*`` tensors with the trained ones,
  3. writing the base ``config.json`` (VL arch) + preprocessors + tokenizer,
  4. saving sharded safetensors with a proper index.

Usage:
  python convert_to_vllm.py \
      --base  /mnt/.../trash/models/Qwen3.5-2B \
      --trained /mnt/.../outputs_toolcalling_sft_v3/qwen35_2b \
      --out   /mnt/.../outputs_toolcalling_sft_v3/qwen35_2b_vllm
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

LANG_PREFIX = "model.language_model."
MAX_SHARD = 5 * 1024 ** 3  # 5 GiB


def load_safetensors_dir(d: Path) -> dict:
    """Load every tensor from a model dir (single file or sharded with an index)."""
    tensors: dict = {}
    idx_path = d / "model.safetensors.index.json"
    single_path = d / "model.safetensors"
    if idx_path.exists():
        weight_map = json.loads(idx_path.read_text())["weight_map"]
        for fname in sorted(set(weight_map.values())):
            with safe_open(d / fname, framework="pt") as fh:
                for key in fh.keys():
                    tensors[key] = fh.get_tensor(key)
    elif single_path.exists():
        with safe_open(single_path, framework="pt") as fh:
            for key in fh.keys():
                tensors[key] = fh.get_tensor(key)
    else:
        raise FileNotFoundError(f"No model.safetensors or index in {d}")
    return tensors


def save_sharded(state_dict: dict, out_dir: Path) -> None:
    """Save tensors to model.safetensors (single) or sharded with an index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sized = [(k, v, v.numel() * v.element_size()) for k, v in state_dict.items()]

    shards: list[dict] = []
    cur: dict = {}
    cur_size = 0
    for key, tensor, nbytes in sized:
        if cur and cur_size + nbytes > MAX_SHARD:
            shards.append(cur)
            cur = {}
            cur_size = 0
        cur[key] = tensor
        cur_size += nbytes
    if cur:
        shards.append(cur)

    if len(shards) == 1:
        for path in out_dir.glob("model*.safetensors*"):
            path.unlink()
        save_file(shards[0], str(out_dir / "model.safetensors"), metadata={"format": "pt"})
        return

    for path in out_dir.glob("model-*.safetensors"):
        path.unlink()
    weight_map: dict[str, str] = {}
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(out_dir / fname), metadata={"format": "pt"})
        for key in shard:
            weight_map[key] = fname
    index = {
        "metadata": {"total_size": sum(nbytes for _, _, nbytes in sized)},
        "weight_map": weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


def copy_aux(base: Path, out: Path) -> None:
    """Copy config + preprocessors + tokenizer from the base VL model dir."""
    keep = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "special_tokens_map.json",
    ]
    for name in keep:
        src = base / name
        if src.exists():
            shutil.copy2(src, out / name)


def copy_training_contract(trained: Path, out: Path) -> None:
    """Make the serving checkpoint self-contained for tool-only inference."""
    for name in ("chat_template.jinja", "toolonly_contract.json"):
        src = trained / name
        if not src.is_file():
            raise FileNotFoundError(
                f"trained checkpoint is missing required contract file: {src}"
            )
        shutil.copy2(src, out / name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True,
                    help="Base VL model dir (e.g. .../trash/models/Qwen3.5-2B).")
    ap.add_argument("--trained", type=Path, required=True,
                    help="Text-only SFT output dir (contains model.safetensors[/index]).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output vLLM-servable (VL-format) dir.")
    args = ap.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        print(f"WARNING: --out {args.out} is not empty; weights will be overwritten.")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading base VL weights: {args.base}")
    base = load_safetensors_dir(args.base)
    print(f"  base tensors: {len(base)}")

    print(f"Loading trained weights: {args.trained}")
    trained = load_safetensors_dir(args.trained)
    trained_lang = {k: v for k, v in trained.items() if k.startswith(LANG_PREFIX)}
    print(f"  trained tensors: {len(trained)}  (language_model: {len(trained_lang)})")

    base_lang_keys = {k for k in base if k.startswith(LANG_PREFIX)}
    missing = base_lang_keys - set(trained_lang)
    extra = set(trained_lang) - base_lang_keys
    if missing or extra:
        print("FATAL: trained language_model keys do not match the base.")
        if missing:
            print(f"  missing from trained ({len(missing)}), e.g.: {sorted(missing)[:5]}")
        if extra:
            print(f"  unexpected in trained ({len(extra)}), e.g.: {sorted(extra)[:5]}")
        return 1

    merged = dict(base)
    merged.update(trained_lang)
    print(f"Merged: {len(merged)} tensors "
          f"(overwrote {len(trained_lang)} language_model tensors with trained values).")

    copy_aux(args.base, args.out)
    # Override the base chat template with the exact template saved by SFT and
    # retain its prompt/template hashes beside the serving weights.
    copy_training_contract(args.trained, args.out)
    save_sharded(merged, args.out)

    arch = json.loads((args.base / "config.json").read_text()).get("architectures")
    print(f"\nDone. VL-format serving checkpoint -> {args.out}")
    print(f"  architectures: {arch}")
    prefixes: dict[str, int] = {}
    for k in merged:
        head = ".".join(k.split(".")[:2])
        prefixes[head] = prefixes.get(head, 0) + 1
    print(f"  tensor prefixes: {prefixes}")
    print("Serve with vLLM via the VL path (Qwen3_5ForConditionalGeneration), "
          "tool-call parser qwen3_coder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
