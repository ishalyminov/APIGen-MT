# SFT, interactive pass@k and offline GSPO

This repository now contains the reusable code from the Qwen3.5 tool-calling
experiments.  Large corpora, rollout archives and checkpoints remain external.

## 1. SFT

Core entrypoints:

- `training/train_toolcalling_toolonly.py` — expands each source trajectory
  into one golden-history next-action example per sequential action, complete
  parallel group, and terminal/no-tool stop; renders with thinking disabled;
  masks loss to exactly one assistant decision; drops rather than truncates an
  over-length prefix.
- `training/convert_to_vllm.py` — converts the trained text checkpoint into the
  Qwen VL serving layout used by vLLM.
- `sweep/submit_sft_sweep.py` and `sweep/run_one_sft.sh` — immutable one-epoch
  learning-rate sweep submission and execution.
- `training/build_next_action_sft_view.py` and
  `training/combine_next_action_corpora.py` — signed, non-destructive corpus
  projection/combination.

Shared train-time surfaces are in `prompts/tool_only_system.txt`,
`templates/qwen35_toolonly_base.jinja`, `data/tools_openai_format.json`, and
`bfcl_eval/tool_schema.py`.

The latest real SFT run used source commit `71cbd318`, one epoch,
`max_length=12288`, seed 42 and all repeat factors equal to one.  Its 1,391
source rows expanded to 13,842 train and 1,466 validation decision prefixes
with zero drops.  The corpus SHA-256 is
`5bf431f0990c7d89288e111dcf37365d9b600be7914ef6e640834cb5735975cc`.

See `examples/rendered_sft/` for literal, auditable strings from that training
split and `training/render_sft_samples.py` for exact reproduction.

## 2. Interactive pass@k

The production evaluator and concrete local-vLLM sweep launcher live in
`evaluation/passk/`.  See its README for the protocol, command line, supported
multi-turn/parallel/refusal behavior and score semantics.

## 3. Offline GSPO

`offline_gspo/` contains rollout admission/reconstruction, the sequence-level
GSPO loss, model helpers, frozen-reference log-probability support, training,
checkpoint recovery, sweep launchers, tests and the pinned Qwen3.5-2B base
contract.  The real jobs were staged from commit `78a44af`.

The recorded replay contained 41,406 episodes in 1,294 homogeneous pass@32
groups (53,534 per-turn segments).  A group was admitted only when its policy
solved the task 1–25 times out of 32.  Rewards are binary, group advantages are
sample-standardized, and GSPO forms one length-normalized ratio over all sampled
assistant tokens in a complete episode.  See `offline_gspo/README.md` for exact
clipping and archival limitations.

## Provenance and exclusions

The SFT/GSPO source snapshot came from local commit `7f4ff42`; the pass@k v8
checker came from the current `tool_synth` experiment tree.  Only code, small
contracts and five rendered examples are committed.  The 180 MiB SFT JSONL,
prepared replay, pass@k rollouts and model checkpoints are intentionally not
included.
