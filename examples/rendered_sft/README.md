# Literal rendered SFT inputs

The five `.txt` files in this directory are not summaries or hand-written
illustrations.  They are byte-for-byte strings returned by the production
trainer's `render()` call for real training-split prefixes from the final
1,391-row targeted-200 corpus.

The trainer then tokenized each string with `add_special_tokens=False`.  The
model receives those token IDs; labels are `-100` outside the token spans listed
in `manifest.json`.  Therefore the `.txt` file is the closest literal readable
representation of what entered the model, while the manifest records the
precise supervised target.

Samples cover:

1. a no-tool stop;
2. a valid recovery call after an earlier no-tool turn;
3. the separate terminal stop after that recovery;
4. a later call conditioned on golden user/tool history;
5. one assistant action containing four independent parallel calls.

They were rendered with the Qwen3.5-2B tokenizer, the committed
`templates/qwen35_toolonly_base.jinja`, `enable_thinking=False`, and the exact
tool-only system prompt.  `manifest.json` pins the corpus, source rows,
template, prompt, catalog, tokenizer files, token counts, masks and rendered
file hashes.  `source_corpus.manifest.json` is the small signed manifest for the
full 180 MiB corpus; the corpus itself is intentionally not committed.

Reproduce them on the original server assets:

```bash
/home/jovyan/.mlspace/envs/qwen35/bin/python \
  training/render_sft_samples.py \
  --corpus /mnt/shared_ru.ml.SZ-5_000264/gambashidze/qwen35_toolonly_sft_sweep_artifacts/data/apigen_toolonly_sft_next_action_targeted200_v1.jsonl \
  --model /mnt/shared_ru.ml.SZ-5_000264/gambashidze/models/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
```

The renderer has fixed token-layout assertions, so tokenizer/template drift is
reported instead of silently producing a different example.
