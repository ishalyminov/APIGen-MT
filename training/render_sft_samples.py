#!/usr/bin/env python3
"""Render a tiny, auditable sample of the exact Qwen SFT input surface.

The output ``.txt`` files contain the unmodified string returned by the same
``render`` function used by ``train_toolcalling_toolonly.py``.  The sidecar
manifest records the exact source row/prefix, tokenizer inputs, loss-mask span,
and content hashes.  It deliberately does not add visual annotations to the
rendered text because those annotations would no longer be what the model saw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

from train_toolcalling_toolonly import (
    build_training_examples_with_tools,
    classify_training_behavior,
    load_catalog,
    load_records,
    render,
    tokenize_with_mask,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data" / "tools_openai_format.json"
DEFAULT_PROMPT = ROOT / "prompts" / "tool_only_system.txt"
DEFAULT_TEMPLATE = ROOT / "templates" / "qwen35_toolonly_base.jinja"
DEFAULT_OUTPUT = ROOT / "examples" / "rendered_sft"

# These are deterministic training-split prefixes from the final 1,391-row
# targeted-200 corpus.  Together they expose no-call, recovery, golden-history,
# terminal-stop and unordered-parallel rendering.
DEFAULT_SAMPLES = (
    (1191, 0, "no_tool_stop"),
    (1191, 1, "recovery_tool_call"),
    (1191, 2, "terminal_stop_after_recovery"),
    (1269, 3, "golden_history_tool_call"),
    (854, 0, "parallel_four_calls"),
)

EXPECTED_TOKEN_LAYOUT = {
    (1191, 0): (476, ((474, 475),)),
    (1191, 1): (555, ((510, 554),)),
    (1191, 2): (578, ((576, 577),)),
    (1269, 3): (857, ((841, 856),)),
    (854, 0): (1316, ((1241, 1315),)),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contiguous_spans(mask: Iterable[int]) -> tuple[tuple[int, int], ...]:
    positions = [index for index, value in enumerate(mask) if value]
    if not positions:
        return ()
    spans: list[tuple[int, int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            spans.append((start, previous + 1))
            start = position
        previous = position
    spans.append((start, previous + 1))
    return tuple(spans)


def source_line_hash(path: Path, row_index: int) -> str:
    position = -1
    with path.open("rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            position += 1
            if position == row_index:
                return sha256_bytes(raw_line.rstrip(b"\r\n"))
    raise IndexError(f"source row {row_index} is missing")


def training_split_indices(num_records: int, *, seed: int = 42) -> set[int]:
    indexed = list(range(num_records))
    random.Random(seed).shuffle(indexed)
    validation_size = max(1, int(round(num_records * 0.1)))
    return set(indexed[validation_size:])


def target_tool_names(
    messages: list[dict[str, Any]], supervised: list[bool]
) -> list[str]:
    names: list[str] = []
    for message, is_target in zip(messages, supervised):
        if not is_target:
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tools-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--chat-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--max-length", type=int, default=12288)
    parser.add_argument(
        "--skip-known-assertions",
        action="store_true",
        help="Allow a different tokenizer/template instead of verifying the recorded run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus = args.corpus.resolve()
    model = args.model.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, load_stats = load_records(corpus)
    training_indices = training_split_indices(len(records))
    catalog = load_catalog(args.tools_catalog)
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    template = args.chat_template.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.chat_template = template

    samples: list[dict[str, Any]] = []
    for row_index, example_index, label in DEFAULT_SAMPLES:
        if row_index not in training_indices:
            raise ValueError(f"row {row_index} is not in the seed-42 training split")
        record = records[row_index]
        examples = build_training_examples_with_tools(
            record, system_prompt, catalog
        )
        try:
            messages, supervised, tools = examples[example_index]
        except IndexError as exc:
            raise IndexError(
                f"row {row_index} has no expanded example {example_index}"
            ) from exc
        rendered = render(
            tokenizer, messages, tools, add_generation_prompt=False
        )
        features = tokenize_with_mask(
            tokenizer,
            messages,
            tools,
            supervised,
            max_length=args.max_length,
            trace_id=f"row{row_index}.example{example_index}",
        )
        if features is None:
            raise ValueError(f"selected sample {row_index}:{example_index} was dropped")

        spans = contiguous_spans(features["completion_mask"])
        expected = EXPECTED_TOKEN_LAYOUT[(row_index, example_index)]
        if not args.skip_known_assertions and (
            features["n_tokens"], spans
        ) != expected:
            raise ValueError(
                f"tokenizer/template drift for {row_index}:{example_index}: "
                f"got {(features['n_tokens'], spans)}, expected {expected}"
            )

        target_ids = [
            token_id
            for token_id, selected in zip(
                features["input_ids"], features["completion_mask"]
            )
            if selected
        ]
        filename = f"{len(samples) + 1:02d}_{label}.txt"
        rendered_path = output_dir / filename
        rendered_path.write_text(rendered, encoding="utf-8")
        samples.append(
            {
                "label": label,
                "rendered_file": filename,
                "rendered_sha256": sha256_file(rendered_path),
                "source_row_index_zero_based": row_index,
                "expanded_example_index_zero_based": example_index,
                "source_line_sha256": source_line_hash(corpus, row_index),
                "split": "train",
                "split_seed": 42,
                "behavior": classify_training_behavior(messages, supervised),
                "visible_tool_count": len(tools),
                "target_tool_names": target_tool_names(messages, supervised),
                "input_token_count": features["n_tokens"],
                "supervised_token_count": features["n_supervised"],
                "supervised_token_spans_half_open": [list(span) for span in spans],
                "supervised_target_text": tokenizer.decode(
                    target_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "input_ids_sha256": sha256_bytes(
                    json.dumps(
                        features["input_ids"], separators=(",", ":")
                    ).encode("utf-8")
                ),
            }
        )

    model_hashes = {}
    for filename in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        path = model / filename
        if path.is_file():
            model_hashes[filename] = sha256_file(path)

    manifest = {
        "format": "exact-rendered-qwen35-toolonly-sft-samples-v1",
        "description": (
            "Each .txt is the literal unannotated output of the production "
            "SFT render() call before tokenization."
        ),
        "corpus": str(corpus),
        "corpus_sha256": sha256_file(corpus),
        "source_manifest": str(corpus.with_suffix(".manifest.json")),
        "source_rows_loaded": len(records),
        "load_statistics": load_stats,
        "model": str(model),
        "model_file_hashes": model_hashes,
        "system_prompt": str(args.system_prompt.resolve()),
        "system_prompt_stripped_content_sha256": sha256_bytes(
            system_prompt.encode("utf-8")
        ),
        "chat_template": str(args.chat_template.resolve()),
        "chat_template_sha256": sha256_file(args.chat_template),
        "tools_catalog": str(args.tools_catalog.resolve()),
        "tools_catalog_sha256": sha256_file(args.tools_catalog),
        "render_contract": {
            "apply_chat_template_tokenize": False,
            "add_generation_prompt": False,
            "enable_thinking": False,
            "tokenizer_add_special_tokens": False,
            "max_length": args.max_length,
            "supervision": "next_action_group_and_terminal_stop",
            "golden_history": True,
        },
        "samples": samples,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(samples)} exact renders to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
