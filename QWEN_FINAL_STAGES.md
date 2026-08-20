# Optional Qwen final-response and grounding stages

This patch separates two late-stage model roles from the main generator and
semantic judge:

- final-response writer;
- final-response grounding judge.

Without any new flags, behavior is unchanged:

```text
final-response writer = --model
final grounding judge = --judge-model, or --model when no judge is set
```

## In-cluster Qwen shortcut

The cluster proxy is expected to expose:

```text
LLM_PROXY_URL=https://176.108.242.226/v1
LLM_PROXY_MASTER_KEY=<secret>
model=Qwen/Qwen3.6-35B-A3B-FP8
```

In the sweethome environment, the URL and key are already exported. Enable both
late-stage roles with one flag:

```bash
python src/generate_step_by_step.py \
  --mode multi-turn \
  --model '<teacher-generator-model>' \
  --judge-model '<teacher-judge-model>' \
  --use-qwen-final-stages \
  ...
```

The resulting five-turn clean route remains nine model requests:

```text
1 teacher blueprint proposal
1 teacher semantic blueprint judge
5 teacher turn compilers
1 local Qwen final-response writer
1 local Qwen grounding judge
```

Only the provider/cost routing changes. Candidate call budgets still count all
nine HTTP attempts across all clients.

## Fine-grained routing

Route only the response writer to Qwen while keeping grounding on the teacher:

```bash
python src/generate_step_by_step.py \
  --final-response-model Qwen/Qwen3.6-35B-A3B-FP8 \
  --final-response-api-base "$LLM_PROXY_URL" \
  --final-response-api-key "$LLM_PROXY_MASTER_KEY" \
  ...
```

Route only grounding to Qwen:

```bash
python src/generate_step_by_step.py \
  --grounding-model Qwen/Qwen3.6-35B-A3B-FP8 \
  --grounding-api-base "$LLM_PROXY_URL" \
  --grounding-api-key "$LLM_PROXY_MASTER_KEY" \
  ...
```

Use arbitrary OpenAI-compatible models/endpoints with the same role-specific
arguments.

## Model routing metadata

Every accepted record now stores model names under:

```json
{
  "generation_metadata": {
    "model_routing": {
      "generator": "...",
      "semantic_judge": "...",
      "final_response_writer": "Qwen/Qwen3.6-35B-A3B-FP8",
      "grounding_judge": "Qwen/Qwen3.6-35B-A3B-FP8"
    }
  }
}
```

Endpoints and API keys are never serialized.

The aggregate usage report also stores `final_response_model` and
`grounding_model`, while request/token budgets include all distinct clients.

## Compatibility details

- The local proxy uses the normal OpenAI `/chat/completions` interface.
- No TLS bypass is added; the environment's trusted certificate path is used.
- OpenRouter-specific `provider` and `reasoning` request fields are omitted for
  role clients pointed at non-OpenRouter endpoints.
- If `--use-qwen-final-stages` is supplied without proxy environment variables,
  startup fails before generation.
- Refusal examples with a deterministic certified native response may bypass the
  normal final-response writer; ordinary final/terminal answers use the routed
  clients.
