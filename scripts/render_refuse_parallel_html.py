#!/usr/bin/env python3
"""Render generated refusal/parallel trajectories as a self-contained HTML file.

All trajectory rows are rendered into the HTML itself. Opening the file directly
does not require a web server, JavaScript, or permission to fetch local JSONL.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def escaped(value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
    return html.escape(text)


def json_block(value: Any, css_class: str = "") -> str:
    return f'<pre class="{css_class}">{escaped(value)}</pre>'


def details(title: str, value: Any, *, open_: bool = False) -> str:
    opened = " open" if open_ else ""
    return (
        f"<details{opened}><summary>{html.escape(title)}</summary>"
        f"{json_block(value)}</details>"
    )


def transition_map(row: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    transitions = (
        row.get("generation_metadata", {})
        .get("evaluation_spec", {})
        .get("transitions", [])
    )
    return {
        (int(item.get("turn_number", 1)), int(item.get("step_number", 1))): item
        for item in transitions
    }


def mode_label(step: dict[str, Any], transition: dict[str, Any]) -> str:
    return str(
        transition.get(
            "mode",
            step.get(
                "execution_mode",
                "parallel" if len(step.get("tool_calls", [])) > 1 else "sequential",
            ),
        )
    )


def render_call(call: dict[str, Any], index: int) -> str:
    name = str(call.get("tool_name", call.get("name", "unknown")))
    return f"""
      <div class="call">
        <div class="call-title">Call {index}: <code>{html.escape(name)}</code></div>
        <div class="two-col">
          <section><h5>Arguments</h5>{json_block(call.get("arguments", {}))}</section>
          <section><h5>Real simulated output</h5>{json_block(call.get("output"))}</section>
        </div>
      </div>
    """


def render_step(
    step: dict[str, Any],
    *,
    turn_number: int,
    transition: dict[str, Any],
) -> str:
    step_number = int(step.get("step_number", 1))
    mode = mode_label(step, transition)
    order_matters = transition.get(
        "call_order_matters", step.get("call_order_matters", True)
    )
    calls = step.get("tool_calls", [])
    calls_html = "".join(
        render_call(call, index) for index, call in enumerate(calls, 1)
    )
    certificate = step.get("quality_verification", {})
    matching = transition.get("matching", {})
    policy_messages = transition.get("policy_messages", [])
    targets = {
        "internal_target": transition.get("internal_target"),
        "bfcl_native_target": transition.get("bfcl_native_target"),
    }
    return f"""
    <article class="step mode-{html.escape(mode)}">
      <div class="step-head">
        <h4>Turn {turn_number}, action {step_number}</h4>
        <span class="pill">{html.escape(mode)}</span>
        <span class="pill {'ordered' if order_matters else 'unordered'}">
          {'ordered' if order_matters else 'unordered within this action'}
        </span>
        <span class="pill">{len(calls)} call{'s' if len(calls) != 1 else ''}</span>
      </div>
      {calls_html}
      {details("Transition quality / feature certificate", certificate)}
      {details("Exact policy-visible messages at this action", policy_messages)}
      {details("Gold targets (internal and BFCL-native)", targets)}
      {details("Matching semantics", matching)}
    </article>
    """


def render_turn(
    turn: dict[str, Any],
    *,
    transitions: dict[tuple[int, int], dict[str, Any]],
) -> str:
    number = int(turn.get("turn_number", 1))
    steps = turn.get("steps", [])
    steps_html = "".join(
        render_step(
            step,
            turn_number=number,
            transition=transitions.get(
                (number, int(step.get("step_number", 1))), {}
            ),
        )
        for step in steps
    )
    return f"""
    <section class="turn">
      <div class="turn-head">
        <h3>Turn {number}</h3>
        <span>{len(steps)} action step{'s' if len(steps) != 1 else ''}</span>
      </div>
      <div class="message user"><b>User</b>{json_block(turn.get("user_query", ""))}</div>
      <div class="expected"><b>Blueprint expected tools:</b>
        {html.escape(", ".join(map(str, turn.get("expected_tools", []))))}
      </div>
      {steps_html}
      <div class="message assistant"><b>Assistant response</b>
        {json_block(turn.get("assistant_response", ""))}
      </div>
      {details("Turn quality verification", turn.get("quality_verification", {}))}
      {details("Safe execution-context marker", turn.get("execution_context", {}))}
    </section>
    """


def render_row(label: str, row: dict[str, Any], index: int) -> str:
    conversation = row.get("conversation")
    transitions = transition_map(row)
    if isinstance(conversation, dict):
        turns = conversation.get("turns", [])
        body = "".join(
            render_turn(turn, transitions=transitions) for turn in turns
        )
        overall_task = conversation.get("overall_task", "")
        initial_state = conversation.get(
            "initial_api_state", row.get("initial_api_state")
        )
        tool_names = conversation.get("tools_used", [])
    else:
        trajectory = row.get("trajectory", {})
        synthetic_turn = {
            "turn_number": 1,
            "user_query": trajectory.get("query", ""),
            "steps": trajectory.get("steps", []),
            "assistant_response": trajectory.get("final_response", ""),
            "expected_tools": row.get("generation_metadata", {}).get(
                "expected_tools", []
            ),
            "quality_verification": {},
            "execution_context": {},
        }
        turns = [synthetic_turn]
        body = render_turn(synthetic_turn, transitions=transitions)
        overall_task = trajectory.get("query", "")
        initial_state = trajectory.get(
            "initial_api_state", row.get("initial_api_state")
        )
        tool_names = trajectory.get("tools_used", [])

    steps = sum(len(turn.get("steps", [])) for turn in turns)
    feature_modes = sorted(
        {
            item.get("mode")
            for item in transitions.values()
            if item.get("mode") in {"refusal", "clarification", "parallel"}
        }
    )
    metadata = row.get("generation_metadata", {})
    verification = row.get("verification_result", {})
    return f"""
    <article class="trajectory" id="{html.escape(label)}-{index}">
      <header class="trajectory-head">
        <div>
          <div class="eyebrow">{html.escape(label)}</div>
          <h2>Trajectory {index}</h2>
        </div>
        <div class="stats">
          <span>{len(turns)} turns</span><span>{steps} steps</span>
          <span>{html.escape(", ".join(feature_modes) or "no feature")}</span>
        </div>
      </header>
      <section class="overview">
        <h3>Actual overall task</h3>
        {json_block(overall_task)}
        <p><b>Tools used:</b> {html.escape(", ".join(map(str, tool_names)))}</p>
      </section>
      {body}
      <section class="raw">
        <h3>Generator and API context</h3>
        {details("Initial API state (generator-only; not policy-visible)", initial_state)}
        {details("Available policy tool schemas", row.get("available_tools", []))}
        {details("Generation metadata and full evaluation spec", metadata)}
        {details("Overall verification result", verification)}
        {details("Complete raw JSON row", row)}
      </section>
    </article>
    """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=JSONL",
        help="May be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Refusal and parallel trajectories")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets: list[tuple[str, Path, list[dict[str, Any]]]] = []
    for raw in args.input:
        if "=" not in raw:
            raise ValueError(f"--input must be LABEL=JSONL, got: {raw}")
        label, path_text = raw.split("=", 1)
        path = Path(path_text)
        datasets.append((label, path, list(read_jsonl(path))))

    cards = "".join(
        f'<div class="card"><b>{html.escape(label)}</b><span>{len(rows)} rows</span>'
        f"<small>{html.escape(str(path))}</small></div>"
        for label, path, rows in datasets
    )
    rows_html = "".join(
        render_row(label, row, index)
        for label, _, rows in datasets
        for index, row in enumerate(rows, 1)
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(args.title)}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0a0d12; --panel:#121722; --line:#2a3343;
      --text:#edf2f7; --muted:#9cabbe; --green:#43d19e; --blue:#72a7ff;
      --orange:#ffb45e; --red:#ff7f8d; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text);
      font:15px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif; }}
    main {{ max-width:1500px; margin:auto; padding:32px 24px 80px; }}
    h1,h2,h3,h4,h5 {{ margin:0 0 10px; line-height:1.2; }}
    h1 {{ font-size:36px; }} h2 {{ font-size:25px; }} h3 {{ font-size:19px; }}
    code,pre {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }}
    pre {{ white-space:pre-wrap; word-break:break-word; margin:8px 0 0;
      padding:12px; background:#090c11; border:1px solid #202938; border-radius:8px; }}
    .lede {{ color:var(--muted); max-width:900px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
      gap:12px; margin:24px 0 36px; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
      padding:16px; display:grid; gap:4px; }}
    .card span {{ color:var(--green); }} .card small {{ color:var(--muted); word-break:break-all; }}
    .trajectory {{ border:1px solid var(--line); border-radius:16px; overflow:hidden;
      margin:0 0 34px; background:#0e131c; }}
    .trajectory-head {{ padding:20px; display:flex; justify-content:space-between;
      gap:20px; align-items:center; background:#151c28; border-bottom:1px solid var(--line); }}
    .eyebrow {{ color:var(--blue); font-weight:700; text-transform:uppercase; letter-spacing:.08em; }}
    .stats {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }}
    .stats span,.pill {{ border:1px solid var(--line); border-radius:999px; padding:4px 9px;
      color:var(--muted); font-size:13px; }}
    .overview,.raw {{ padding:20px; }}
    .turn {{ margin:0 20px 24px; border:1px solid var(--line); border-radius:12px;
      padding:16px; background:var(--panel); }}
    .turn-head,.step-head {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
    .turn-head {{ justify-content:space-between; border-bottom:1px solid var(--line);
      padding-bottom:10px; margin-bottom:14px; }}
    .turn-head span,.expected {{ color:var(--muted); }}
    .message {{ border-left:3px solid var(--blue); margin:12px 0; padding:10px 12px; background:#101722; }}
    .message.assistant {{ border-left-color:var(--green); }}
    .step {{ margin:14px 0; padding:14px; border:1px solid var(--line); border-radius:10px; }}
    .mode-parallel {{ border-color:#7556c7; }}
    .mode-refusal,.mode-clarification {{ border-color:#a95462; }}
    .mode-parallel .pill:first-of-type {{ color:#c5abff; }}
    .mode-refusal .pill:first-of-type,.mode-clarification .pill:first-of-type {{ color:var(--red); }}
    .ordered {{ color:var(--orange); }} .unordered {{ color:#c5abff; }}
    .call {{ margin:12px 0; padding:12px; border-radius:9px; background:#0c1119; }}
    .call-title {{ font-weight:700; }} .call-title code {{ color:var(--green); }}
    .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:10px; }}
    h5 {{ color:var(--muted); }}
    details {{ margin:10px 0; border-top:1px solid #222b39; padding-top:8px; }}
    summary {{ cursor:pointer; color:var(--blue); font-weight:600; }}
    @media (max-width:800px) {{ .two-col {{ grid-template-columns:1fr; }}
      .trajectory-head {{ align-items:flex-start; flex-direction:column; }}
      .stats {{ justify-content:flex-start; }} }}
  </style>
</head>
<body><main>
  <h1>{html.escape(args.title)}</h1>
  <p class="lede">Static self-contained review file. Every trajectory row is
  present in the HTML source and visible without JavaScript. Tool outputs are
  simulator outputs; initial API state is explicitly labeled generator-only.</p>
  <div class="cards">{cards}</div>
  {rows_html}
</main></body></html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(
        f"Wrote {sum(len(rows) for _, _, rows in datasets)} trajectories "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
