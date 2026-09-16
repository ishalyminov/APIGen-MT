# Interactive APIGen pass@k

`check_apigen_trajectories_passk_v3.py` is the evaluator used for the latest
reasoning and offline-GSPO experiments.  Despite the historical filename, its
protocol is `apigen-semantic-reasoning-passk-v8`.

It evaluates the policy interactively.  After an exact currently-ready action,
the environment replays the recorded gold tool output and lets the model choose
the next action.  It supports:

- real multi-turn boundaries and per-turn tool snapshots;
- dependent multi-step chains and ordered mutation barriers;
- batching/reordering of currently-ready independent reads;
- explicit parallel steps compared as unordered multisets (duplicates still
  count);
- terminal/no-tool decisions and refusal-shaped rows;
- hidden-state and hidden-future-output exclusion.

The primary episode score rejects a tool call when the gold decision is
no-tool, but permits visible terminal prose.  The summary also reports
`strict_tool_only_metrics`, which requires the visible output to be empty.

## Run against an OpenAI-compatible endpoint

```bash
python evaluation/passk/check_apigen_trajectories_passk_v3.py \
  --jsonl /path/to/apigen_next_action.jsonl \
  --tool-pool magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl \
  --out-dir /path/to/results \
  --tool-scope declared \
  --pass-k 16 \
  --temperature 1.0 \
  --max-tokens 8192 \
  --workers 32 \
  --enable-thinking \
  --chat-template-path templates/qwen35_toolonly_base.jinja \
  --vllm-url http://127.0.0.1:8000/v1 \
  --model qwen3.5-2b \
  --resume
```

The JSONL must have its adjacent `.manifest.json`; the evaluator fail-closes on
the corpus hash/supervision contract and, when supplied, the serving template
hash.  `merge_apigen_passk_v3_shards.py` merges completed shards only after
re-validating them.

`run_qwen35_2b_4b_reasoning_pass32.py` is the concrete four-GPU launcher used
for the 2B/4B, temperature 0.7/1.0 pass@32 experiment.  Its model/runtime paths
are server-specific, while its checker, template and tool-pool paths resolve
inside this repository.

The reasoning prompt here intentionally differs from the SFT tool-only prompt:
pass@k was sampled with thinking enabled, whereas SFT renders use
`enable_thinking=False`.  The chat template is shared.

## Test

```bash
pytest -q evaluation/passk/tests
```

The stale teacher-forced partial evaluator is deliberately not included as a
supported entrypoint; it needs its refusal projection updated before reuse.
