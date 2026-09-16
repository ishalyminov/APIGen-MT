# Offline GSPO on the Qwen3.5-2B pass@32 archive

This experiment uses only homogeneous pass@32 groups from the reasoning-on
Qwen3.5-2B run. A `(task, temperature)` group is admitted when the model solved
it at least once and fewer than 80% of the time: 1–25 successes out of 32.
Temperature 0.7 and 1.0 are separate behavior-policy groups. The one archived
API-error episode is dropped. One additional, digest-pinned length response is
dropped because its decoded text re-encodes to 8,191 tokens while vLLM recorded
8,192 and the archive did not retain raw token IDs. Group advantages are
recomputed after both exclusions; every reconstructable policy failure remains
a reward-0 episode.

The loss is GSPO, not token-wise GRPO. For each complete interactive episode,
all sampled assistant tokens across all reached user turns contribute to one
length-normalized sequence ratio. The ratio is clipped asymmetrically at
`[1 - 3e-4, 1 + 4e-4]` and multiplied by the sample-standardized binary
group advantage. One complete pass@32 group is one optimizer update.

Important archival limitation: successful native calls were stored as parsed
arguments rather than exact XML bytes, and sampling-time token logprobs were
not stored. The replay therefore canonically re-renders parsed calls and uses
temperature-scaled full-softmax probabilities; `top_k=20/top_p=.95` truncation
cannot be reconstructed. The experiment is named **canonical nominal-policy
offline GSPO**, not exact behavior-policy importance sampling.

## Artifacts

- Prepared replay:
  `/mnt/shared_ru.ml.SZ-5_000264/gambashidze/qwen35_toolonly_sft_sweep_artifacts/offline_gspo/q35_2b_pass32_reasoning_v4_gspo_v1/prepared`
- Frozen-reference denominator: recomputed online by a second immutable base
  model in each training process. This avoids the cross-GPU/kernel drift found
  in the earlier cached-logprob attempt.
- Rollout-policy identity: `qwen35_2b_base_contract.json` pins the model config,
  tokenizer files, weight shard, and weight index. The tokenizer hashes must
  also match the prepared replay manifest before either model is loaded.
- Checkpoints:
  `/mnt/shared_ru.ml.SZ-5_000264/gambashidze/qwen35_toolonly_sft_sweep_artifacts/checkpoints/q35-2b-offline-gspo-pass32-v1/`
- BFCL metrics:
  `/mnt/shared_ru.ml.SZ-5_000264/gambashidze/qwen35_toolonly_sft_sweep_artifacts/simple_metrics_q35-2b-offline-gspo-pass32-v1.{json,md}`

## Reproduce

```bash
python -m offline_gspo.prepare_offline_gspo \
  --output-dir "$REPLAY_ROOT/prepared"

python -m offline_gspo.submit_gspo_sweep
```

Use `evaluation/passk/check_apigen_trajectories_passk_v3.py` to evaluate the
resulting serving checkpoints.  The former experiment-specific
`wait_then_eval.py` depended on a separate BFCL checkout and is intentionally
not presented here as part of the reusable GSPO core.

The prepared replay, policy archives and checkpoints are external artifacts;
they are not committed because they are large.  The builder validates their
signed dataset, prompt, template, model and rollout contracts before use.

Training writes an atomic optimizer-boundary checkpoint every 400 groups and
removes that large transient state only after the final model and training
contract are safely written. The legacy precompute scripts remain available
for diagnostics, but the sweep does not consume a cross-runtime cache.
