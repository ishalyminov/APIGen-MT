#!/usr/bin/env python3
"""Build a self-contained browser for APIGen trajectory JSONL files."""

from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def trajectory_view(row: dict[str, Any]) -> dict[str, Any]:
    """Return one display shape for both legacy and batched-turn rows."""

    trajectory = row.get("trajectory")
    if isinstance(trajectory, dict):
        turns = trajectory.get("turns")
        if not isinstance(turns, list):
            turns = [
                {
                    "turn_number": 1,
                    "user_query": trajectory.get("query", ""),
                    "query_intent": trajectory.get("query_intent", ""),
                    "steps": trajectory.get("steps", []),
                    "assistant_response": trajectory.get("final_response", ""),
                }
            ]
        return {
            **trajectory,
            "query": trajectory.get("query", ""),
            "turns": turns,
        }

    conversation = row.get("conversation")
    if not isinstance(conversation, dict):
        return {"query": "", "turns": [], "steps": []}
    turns = conversation.get("turns") or []
    return {
        "query": conversation.get("overall_task", ""),
        "query_intent": conversation.get("overall_task", ""),
        "categories_used": conversation.get("categories_used", []),
        "tools_used": conversation.get("tools_used", []),
        "initial_api_state": conversation.get("initial_api_state", {}),
        "turns": turns,
        "steps": [
            step
            for turn in turns
            for step in (turn.get("steps") or [])
        ],
        "final_response": (
            turns[-1].get("assistant_response", "") if turns else ""
        ),
    }


def first_call(row: dict[str, Any]) -> dict[str, Any]:
    steps = trajectory_view(row).get("steps", [])
    if not steps:
        return {}
    calls = steps[0].get("tool_calls", [])
    return calls[0] if calls else {}


def semantic_signature(row: dict[str, Any]) -> str:
    calls: list[dict[str, Any]] = []
    for step in trajectory_view(row).get("steps", []):
        for call in step.get("tool_calls", []):
            calls.append(
                {
                    "name": call.get("tool_name"),
                    "arguments": call.get("arguments", {}),
                }
            )
    canonical = json.dumps(calls, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def normalise_text(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def grounding_hints(row: dict[str, Any]) -> list[str]:
    """Conservative hints for first-call string values missing from the query."""
    trajectory = trajectory_view(row)
    query = normalise_text(
        "\n".join(
            str(turn.get("user_query") or "")
            for turn in trajectory.get("turns", [])
        )
        or trajectory.get("query", "")
    )
    hints: list[str] = []
    arguments = first_call(row).get("arguments", {})
    for key, value in arguments.items():
        if not isinstance(value, str) or not value.strip():
            continue
        normalised = normalise_text(value)
        if len(normalised) < 2 or normalised in query:
            continue
        hints.append(f'first-call {key}="{value}" is not verbatim in query')
    return hints


def event_summary(
    events: list[dict[str, Any]], num_gold_calls: int
) -> dict[str, Any]:
    failures = collections.Counter(
        event.get("failure") or "none" for event in events
    )
    turns: dict[str, dict[str, int]] = {}
    for turn, turn_events in _group_by(events, lambda event: event.get("turn")).items():
        turns[str(turn)] = {
            "total": len(turn_events),
            "matched": sum(bool(event.get("matched")) for event in turn_events),
            "failed": sum(not bool(event.get("matched")) for event in turn_events),
        }
    failed = [event for event in events if not event.get("matched")]
    matched = [event for event in events if event.get("matched")]
    samples = _group_by(events, lambda event: event.get("sample_index"))
    successful_samples = 0
    for sample_events in samples.values():
        matched_steps = {
            int(step_index)
            for event in sample_events
            if event.get("matched")
            for step_index in event.get("matched_gold_step_indices", [])
        }
        if (
            len(matched_steps) == num_gold_calls
            and all(bool(event.get("matched")) for event in sample_events)
        ):
            successful_samples += 1
    task_unsolved = bool(samples) and successful_samples == 0
    return {
        "total": len(events),
        "matched": len(matched),
        "failed": len(failed),
        "num_samples": len(samples),
        "successful_samples": successful_samples,
        "task_unsolved": task_unsolved,
        "failure_counts": dict(failures),
        "turns": turns,
        "systematic_failure": task_unsolved
        or any(
            counts["total"] >= 8 and counts["matched"] == 0
            for counts in turns.values()
        ),
        "sample_failures": failed[:8],
        "sample_matches": matched[:2],
    }


def _group_by(items: list[Any], key: Any) -> dict[Any, list[Any]]:
    result: dict[Any, list[Any]] = collections.defaultdict(list)
    for item in items:
        result[key(item)].append(item)
    return result


def json_for_script(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def pretty_html(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2))


def static_details(title: str, value: Any, *, open_by_default: bool = False) -> str:
    opened = " open" if open_by_default else ""
    return (
        f"<details{opened}><summary>{html.escape(title)}</summary>"
        f"<pre>{pretty_html(value)}</pre></details>"
    )


def flatten_state(
    value: Any, prefix: str = "", output: dict[str, Any] | None = None
) -> dict[str, Any]:
    output = output if output is not None else {}
    if isinstance(value, dict):
        if not value:
            output[prefix or "$"] = {}
        for key, child in value.items():
            flatten_state(child, f"{prefix}.{key}" if prefix else key, output)
    else:
        output[prefix or "$"] = value
    return output


def static_state_diff(before: Any, after: Any) -> str:
    flattened_before = flatten_state(before)
    flattened_after = flatten_state(after)
    paths = sorted(set(flattened_before) | set(flattened_after))
    changed = [
        path
        for path in paths
        if flattened_before.get(path) != flattened_after.get(path)
    ]
    if not changed:
        return '<div class="meta">No state changes.</div>'
    body = "".join(
        "<tr>"
        f"<td>{html.escape(path)}</td>"
        f'<td class="before">{html.escape(json.dumps(flattened_before.get(path), ensure_ascii=False))}</td>'
        f'<td class="after">{html.escape(json.dumps(flattened_after.get(path), ensure_ascii=False))}</td>'
        "</tr>"
        for path in changed
    )
    return (
        '<table class="diff"><thead><tr><th>Path</th><th>Before</th>'
        f"<th>After</th></tr></thead><tbody>{body}</tbody></table>"
    )


def static_review_html(
    payload: list[dict[str, Any]], summary: dict[str, Any]
) -> tuple[str, str]:
    indices = (
        list(range(len(payload)))
        if len(payload) <= 32
        else summary["curated_indices"] or list(range(32))
    )
    sidebar: list[str] = []
    articles: list[str] = [
        '<div class="notice"><b>Static fallback view.</b> These curated rows are '
        "pre-rendered and work without JavaScript. With JavaScript enabled, the "
        "reviewer upgrades to searchable access across all 500 rows.</div>"
    ]
    for index in indices:
        item = payload[index]
        row = item["row"]
        review = item["review"]
        trajectory = trajectory_view(row)
        query = trajectory.get("query", "")
        categories = ", ".join(review["categories"])
        sidebar.append(
            f'<a class="item" href="#static-row-{index}"><div class="q">'
            f"<b>#{index}</b> {html.escape(query)}</div>"
            f'<div class="meta">{html.escape(categories)}</div></a>'
        )
        notices: list[str] = []
        if review["model_eval"]["systematic_failure"]:
            notices.append(
                '<div class="notice danger"><b>Systematic Qwen failure.</b> '
                "Inspect query versus gold arguments.</div>"
            )
        if review["grounding_hints"]:
            notices.append(
                '<div class="notice"><b>Grounding hints:</b><br>'
                + "<br>".join(html.escape(hint) for hint in review["grounding_hints"])
                + "</div>"
            )
        turns: list[str] = []
        for turn_position, turn in enumerate(trajectory.get("turns", []), 1):
            steps: list[str] = []
            for step_position, step in enumerate(turn.get("steps", []), 1):
                calls = "".join(
                    '<div class="call">'
                    f'<div class="call-name">{html.escape(str(call.get("tool_name")))}</div>'
                    '<div class="grid"><div>'
                    + static_details("Arguments", call.get("arguments"), open_by_default=True)
                    + "</div><div>"
                    + static_details(
                        "Recorded tool output", call.get("output"), open_by_default=True
                    )
                    + "</div></div></div>"
                    for call in step.get("tool_calls", [])
                )
                steps.append(
                    '<section class="card step">'
                    f"<h3>Step {step.get('step_number', step_position)}</h3>{calls}"
                    f'<div class="meta">{html.escape(str(step.get("reasoning", "")))}</div>'
                    "<h3>API state diff</h3>"
                    + static_state_diff(step.get("pre_state"), step.get("post_state"))
                    + static_details("Pre-state", step.get("pre_state"))
                    + static_details("Post-state", step.get("post_state"))
                    + static_details(
                        "State verification",
                        step.get("state_verification"),
                        open_by_default=True,
                    )
                    + static_details(
                        "Quality verification",
                        step.get("quality_verification"),
                        open_by_default=True,
                    )
                    + "</section>"
                )
            turns.append(
                '<section class="turn-block">'
                f"<h2>Turn {turn.get('turn_number', turn_position)}</h2>"
                f'<div class="query">{html.escape(str(turn.get("user_query", "")))}</div>'
                f'<div class="meta">Intent: {html.escape(str(turn.get("query_intent", "")))}</div>'
                + "".join(steps)
                + '<section class="card"><h3>Assistant response</h3>'
                f'<div>{html.escape(str(turn.get("assistant_response", "")))}</div>'
                + static_details("Turn quality verification", turn.get("quality_verification"))
                + "</section></section>"
            )
        articles.append(
            f'<article id="static-row-{index}" class="static-example">'
            f"<h1>Row #{index}</h1>"
            f'<div class="meta">{html.escape(categories)} · '
            f'{html.escape(", ".join(review["tools"]))}</div>'
            + "".join(notices)
            + f'<div class="query">{html.escape(query)}</div>'
            + '<section class="card"><h2>Initial hidden API state</h2>'
            + static_details(
                "initial_api_state", row.get("initial_api_state"), open_by_default=True
            )
            + "</section>"
            + "".join(turns)
            + '<section class="card"><h2>Audit evidence</h2>'
            + static_details("Generation metadata", row.get("generation_metadata"))
            + static_details("Full verification result", row.get("verification_result"))
            + static_details("Available tool schemas", row.get("available_tools"))
            + "</section></article>"
        )
    return "".join(sidebar), "".join(articles)


def build_payload(
    rows: list[dict[str, Any]], events: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events_by_row = _group_by(events, lambda event: event.get("row_position"))
    signatures = [semantic_signature(row) for row in rows]
    signature_counts = collections.Counter(signatures)
    signature_first: dict[str, int] = {}
    for index, signature in enumerate(signatures):
        signature_first.setdefault(signature, index)

    categories: collections.Counter[str] = collections.Counter()
    tools: collections.Counter[str] = collections.Counter()
    payload: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        trajectory = trajectory_view(row)
        row_categories = trajectory.get("categories_used", [])
        row_tools = trajectory.get("tools_used", [])
        categories.update(row_categories)
        tools.update(row_tools)
        hints = grounding_hints(row)
        signature = signatures[index]
        review = {
            "index": index,
            "categories": row_categories,
            "tools": row_tools,
            "semantic_signature": signature,
            "semantic_duplicate_count": signature_counts[signature],
            "semantic_first_index": signature_first[signature],
            "grounding_hints": hints,
            "model_eval": event_summary(
                events_by_row.get(index, []),
                len(trajectory.get("steps", [])),
            ),
        }
        payload.append({"row": row, "review": review})

    systematic_rows = [
        item["review"]["index"]
        for item in payload
        if item["review"]["model_eval"]["systematic_failure"]
    ]
    hinted_rows = [
        item["review"]["index"]
        for item in payload
        if item["review"]["grounding_hints"]
    ]
    curated: list[int] = []
    for index in systematic_rows + hinted_rows:
        if index not in curated:
            curated.append(index)
    per_category: collections.Counter[str] = collections.Counter()
    for item in payload:
        category = (item["review"]["categories"] or ["Uncategorised"])[0]
        if per_category[category] < 3:
            curated.append(item["review"]["index"])
            per_category[category] += 1
    curated = list(dict.fromkeys(curated))

    summary = {
        "rows": len(rows),
        "events": len(events),
        "categories": dict(sorted(categories.items())),
        "tools": dict(tools.most_common()),
        "semantic_unique": len(signature_counts),
        "semantic_duplicate_excess": len(rows) - len(signature_counts),
        "systematic_eval_rows": systematic_rows,
        "grounding_hint_rows": hinted_rows,
        "curated_indices": curated,
    }
    return payload, summary


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE_TEXT__</title>
<style>
:root {
  --bg:#0b0f14; --panel:#111821; --panel2:#151f2b; --line:#263445;
  --text:#e7edf5; --muted:#95a5b8; --accent:#67d8b1; --blue:#72b7ff;
  --warn:#ffca66; --bad:#ff7d7d; --good:#70e19b; --code:#091018;
}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
button,input,select{font:inherit;color:var(--text);background:var(--panel2);border:1px solid var(--line);
border-radius:8px;padding:8px 10px} button{cursor:pointer} button:hover{border-color:var(--accent)}
.app{display:grid;grid-template-rows:auto 1fr;height:100vh}.top{padding:14px 18px;border-bottom:1px solid var(--line);
background:#0e141c;display:flex;gap:12px;align-items:center;flex-wrap:wrap}.brand{font-weight:750;font-size:17px;margin-right:8px}
.stat{color:var(--muted);padding:4px 8px;border:1px solid var(--line);border-radius:999px}.stat b{color:var(--text)}
.layout{display:grid;grid-template-columns:330px minmax(0,1fr);min-height:0}.sidebar{border-right:1px solid var(--line);
padding:12px;overflow:auto;background:#0e141c}.controls{display:grid;gap:8px;margin-bottom:12px}.row{display:flex;gap:8px}.row>*{min-width:0;flex:1}
.list{display:grid;gap:6px}.item{padding:9px;border:1px solid transparent;border-radius:8px;background:var(--panel);
cursor:pointer}.item:hover,.item.active{border-color:var(--accent)}.item .q{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
a.item{display:block;color:var(--text);text-decoration:none}.static-example{padding:12px 0 32px;border-bottom:2px solid var(--line)}
.meta{font-size:12px;color:var(--muted);display:flex;gap:6px;flex-wrap:wrap}.tag{border-radius:999px;padding:1px 6px;background:#1b2937}
.tag.bad{background:#42252a;color:#ffc3c3}.tag.warn{background:#3c321e;color:#ffe0a0}.tag.good{background:#18382b;color:#b8ffd5}
.main{overflow:auto;padding:22px 28px}.content{max-width:1180px;margin:auto}.headline{display:flex;justify-content:space-between;
gap:12px;align-items:flex-start}.headline h1{font-size:22px;margin:0 0 4px}.query{font-size:18px;line-height:1.55;
padding:16px 18px;border-left:4px solid var(--accent);background:var(--panel);border-radius:8px;margin:16px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}.card{background:var(--panel);
border:1px solid var(--line);border-radius:10px;padding:14px;margin:10px 0}.card h2,.card h3{margin:0 0 10px}.step{border-left:3px solid var(--blue)}
.call{background:var(--panel2);padding:12px;border-radius:8px;margin:8px 0}.call-name{font-weight:750;color:var(--blue)}
pre{background:var(--code);border:1px solid #1d2a38;border-radius:8px;padding:12px;overflow:auto;max-height:520px;
white-space:pre-wrap;word-break:break-word;color:#d8e6f5}details{margin:8px 0}summary{cursor:pointer;color:var(--blue);font-weight:650}
.diff{width:100%;border-collapse:collapse;font-size:12px}.diff td,.diff th{border-bottom:1px solid var(--line);padding:6px;text-align:left;vertical-align:top}
.diff .before{color:#ffb1b1}.diff .after{color:#aef0c6}.ok{color:var(--good)}.bad-text{color:var(--bad)}.warn-text{color:var(--warn)}
.notice{padding:11px 13px;border-radius:8px;background:#3c321e;border:1px solid #665126;margin:8px 0}.danger{background:#40232a;border-color:#6f3541}
.turn-block{margin:24px 0 34px;padding-top:8px;border-top:2px solid var(--line)}.turn-block>h2{font-size:20px;color:var(--accent)}
.empty{color:var(--muted);padding:25px;text-align:center}.footer-note{color:var(--muted);font-size:12px;margin-top:18px}
@media(max-width:850px){.layout{grid-template-columns:1fr}.sidebar{max-height:42vh;border-right:0;border-bottom:1px solid var(--line)}
.app{height:auto}.main{padding:16px}.headline{display:block}}
</style>
</head>
<body>
<div class="app">
  <header class="top">
    <div class="brand">APIGen trajectory reviewer</div>
    <span class="stat"><b id="rowCount"></b> rows</span>
    <span class="stat"><b id="uniqueCount"></b> semantic unique</span>
    <span class="stat"><b id="eventCount"></b> Qwen events in snapshot</span>
    <span class="stat">pass@1: <b id="pass1"></b></span>
    <span class="stat">pass@16: <b id="pass16"></b></span>
    <span class="stat">view: <b>full multi-turn trajectory</b></span>
  </header>
  <div class="layout">
    <aside class="sidebar">
      <div class="controls">
        <input id="search" placeholder="Search query, tool, category…">
        <div class="row">
          <select id="category"></select>
          <select id="filter">
            <option value="all">All rows</option>
            <option value="curated">Curated review set</option>
            <option value="eval-failed">Any Qwen failure</option>
            <option value="systematic">Systematic eval failure</option>
            <option value="grounding">Grounding hints</option>
            <option value="duplicates">Semantic duplicates</option>
          </select>
        </div>
        <div class="row">
          <input id="jump" type="number" min="0" placeholder="Row index">
          <button id="jumpButton">Go</button>
          <button id="randomButton">Random</button>
        </div>
        <div class="meta"><span id="visibleCount"></span></div>
      </div>
      <div id="list" class="list">__STATIC_LIST_HTML__</div>
    </aside>
    <main class="main"><div id="content" class="content">__STATIC_CONTENT_HTML__</div></main>
  </div>
</div>
<script>
const DATA=__PAYLOAD_JSON__;
const SUMMARY=__SUMMARY_JSON__;
const $=id=>document.getElementById(id);
const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pretty=value=>esc(JSON.stringify(value??null,null,2));
const badge=(text,kind="")=>`<span class="tag ${kind}">${esc(text)}</span>`;
const detail=(title,value,open=false)=>`<details ${open?"open":""}><summary>${esc(title)}</summary><pre>${pretty(value)}</pre></details>`;
let filtered=DATA.map((_,i)=>i), selected=0;

function trajectoryView(row){
  if(row.trajectory){
    const t=row.trajectory;
    const turns=Array.isArray(t.turns)?t.turns:[{
      turn_number:1,user_query:t.query||"",query_intent:t.query_intent||"",
      steps:t.steps||[],assistant_response:t.final_response||""
    }];
    return {...t,query:t.query||"",turns};
  }
  const c=row.conversation||{},turns=Array.isArray(c.turns)?c.turns:[];
  return {
    query:c.overall_task||"",query_intent:c.overall_task||"",
    categories_used:c.categories_used||[],tools_used:c.tools_used||[],
    initial_api_state:c.initial_api_state||{},turns,
    steps:turns.flatMap(turn=>turn.steps||[]),
    final_response:turns.length?(turns[turns.length-1].assistant_response||""):""
  };
}

function flatten(value,prefix="",out={}){
  if(value && typeof value==="object" && !Array.isArray(value)){
    const keys=Object.keys(value); if(!keys.length) out[prefix||"$"]={};
    for(const key of keys) flatten(value[key],prefix?`${prefix}.${key}`:key,out);
  } else out[prefix||"$"]=value;
  return out;
}
function stateDiff(before,after){
  const a=flatten(before),b=flatten(after),keys=[...new Set([...Object.keys(a),...Object.keys(b)])].sort();
  const changed=keys.filter(k=>JSON.stringify(a[k])!==JSON.stringify(b[k]));
  if(!changed.length)return `<div class="meta">No state changes.</div>`;
  return `<table class="diff"><thead><tr><th>Path</th><th>Before</th><th>After</th></tr></thead><tbody>`+
    changed.map(k=>`<tr><td>${esc(k)}</td><td class="before">${esc(JSON.stringify(a[k]))}</td><td class="after">${esc(JSON.stringify(b[k]))}</td></tr>`).join("")+
    `</tbody></table>`;
}
function renderCall(call){
  return `<div class="call"><div class="call-name">${esc(call.tool_name)}</div>
    <div class="grid"><div>${detail("Arguments",call.arguments,true)}</div><div>${detail("Recorded tool output",call.output,true)}</div></div></div>`;
}
function renderStep(step,index){
  return `<section class="card step"><h3>Step ${esc(step.step_number||index+1)}</h3>
    ${(step.tool_calls||[]).map(renderCall).join("")}
    <div><b>Generator reasoning</b><div class="meta">${esc(step.reasoning||"")}</div></div>
    <h3 style="margin-top:14px">API state diff</h3>${stateDiff(step.pre_state,step.post_state)}
    ${detail("Pre-state",step.pre_state)}${detail("Post-state",step.post_state)}
    <div class="grid"><div>${detail("State verification",step.state_verification,true)}</div>
    <div>${detail("Quality verification",step.quality_verification,true)}</div></div></section>`;
}
function renderTurn(turn,index){
  return `<section class="turn-block"><h2>Turn ${esc(turn.turn_number||index+1)}</h2>
    <div class="query">${esc(turn.user_query||turn.query||"")}</div>
    <div class="meta">Intent: ${esc(turn.query_intent||"")}</div>
    ${(turn.steps||[]).map(renderStep).join("")}
    <section class="card"><h3>Assistant response</h3><div>${esc(turn.assistant_response||"")}</div>
    ${detail("Expected tools",turn.expected_tools||[])}
    ${detail("Execution context",turn.execution_context||{})}
    ${detail("Turn quality verification",turn.quality_verification||{})}</section></section>`;
}
function renderEval(review){
  const ev=review.model_eval;
  if(!ev.total)return `<div class="meta">No Qwen rollout events for this row at snapshot time.</div>`;
  const turns=Object.entries(ev.turns).map(([turn,v])=>`turn ${turn}: ${v.matched}/${v.total} matched`).join(" · ");
  return `<div class="meta">${esc(turns)}</div>
    <div>${Object.entries(ev.failure_counts).map(([k,v])=>badge(`${k}: ${v}`,k==="none"?"good":"bad")).join(" ")}</div>
    ${ev.sample_failures.length?detail("Sample failed predictions",ev.sample_failures):""}
    ${ev.sample_matches.length?detail("Sample matched predictions",ev.sample_matches):""}`;
}
function render(index){
  selected=index; const item=DATA[index],row=item.row,r=item.review,t=trajectoryView(row);
  const gates=row.verification_result||{},meta=row.generation_metadata||{};
  const notices=[];
  if(r.model_eval.systematic_failure)notices.push(`<div class="notice danger"><b>Systematic Qwen failure in snapshot.</b> Inspect query vs gold arguments below.</div>`);
  if(r.grounding_hints.length)notices.push(`<div class="notice"><b>Heuristic grounding hints:</b><br>${r.grounding_hints.map(esc).join("<br>")}</div>`);
  if(r.semantic_duplicate_count>1)notices.push(`<div class="notice"><b>Semantic duplicate group:</b> ${r.semantic_duplicate_count} rows; first row #${r.semantic_first_index}.</div>`);
  const turns=(t.turns||[]).map(renderTurn).join("");
  $("content").innerHTML=`<div class="headline"><div><h1>Row #${index}</h1><div class="meta">
    ${(r.categories||[]).map(x=>badge(x)).join(" ")} ${(r.tools||[]).map(x=>badge(x)).join(" ")}</div></div>
    <div class="meta">attempt ${esc(row.generation_attempt)} · ${esc(row.timestamp)}</div></div>
    ${notices.join("")}<div class="query">${esc(t.query)}</div>
    <div class="grid"><section class="card"><h3>Generation</h3>
      ${badge(meta.rl_quality_gate_passed?"RL gate passed":"RL gate failed",meta.rl_quality_gate_passed?"good":"bad")}
      ${badge(gates.overall_verification_passed?"Verifier passed":"Verifier failed",gates.overall_verification_passed?"good":"bad")}
      <div class="meta">intent: ${esc(meta.query_intent)}</div></section>
      <section class="card"><h3>Qwen proxy snapshot</h3>${renderEval(r)}</section></div>
    <section class="card"><h2>Initial hidden API state</h2>${detail("initial_api_state",row.initial_api_state||t.initial_api_state,true)}</section>
    ${turns}
    <section class="card"><h2>Audit evidence</h2>
      ${detail("Generation metadata",row.generation_metadata)}
      ${detail("Full verification result",row.verification_result)}
      ${detail("Intermediate API states",row.intermediate_api_states)}
      ${detail(`Available tool schemas (${(row.available_tools||[]).length})`,row.available_tools)}
      ${detail("Token usage",row.token_usage)}
    </section>
    <div class="footer-note">Grounding hints are heuristic review aids, not automatic rejection decisions. Qwen events reflect the build-time snapshot.</div>`;
  document.querySelectorAll(".item").forEach(node=>node.classList.toggle("active",Number(node.dataset.index)===index));
  window.location.hash=`row-${index}`;
  document.querySelector(".main").scrollTop=0;
}
function applyFilters(){
  const search=$("search").value.toLowerCase(),cat=$("category").value,mode=$("filter").value;
  const curated=new Set(SUMMARY.curated_indices);
  filtered=DATA.map((item,index)=>({item,index})).filter(({item,index})=>{
    const r=item.review,t=trajectoryView(item.row);
    const hay=[t.query,...r.categories,...r.tools].join(" ").toLowerCase();
    if(search&&!hay.includes(search))return false;
    if(cat&&!(r.categories||[]).includes(cat))return false;
    if(mode==="curated"&&!curated.has(index))return false;
    if(mode==="eval-failed"&&!r.model_eval.failed)return false;
    if(mode==="systematic"&&!r.model_eval.systematic_failure)return false;
    if(mode==="grounding"&&!r.grounding_hints.length)return false;
    if(mode==="duplicates"&&r.semantic_duplicate_count<2)return false;
    return true;
  }).map(x=>x.index);
  $("visibleCount").textContent=`${filtered.length} visible`;
  $("list").innerHTML=filtered.map(index=>{
    const {row,review:r}=DATA[index],t=trajectoryView(row);
    return `<div class="item ${index===selected?"active":""}" data-index="${index}">
      <div class="q"><b>#${index}</b> ${esc(t.query)}</div><div class="meta">
      ${(r.categories||[]).map(x=>badge(x)).join("")}
      ${r.model_eval.failed?badge(`${r.model_eval.failed} eval fail`,"bad"):""}
      ${r.grounding_hints.length?badge("grounding hint","warn"):""}
      ${r.semantic_duplicate_count>1?badge(`dup ×${r.semantic_duplicate_count}`,"warn"):""}</div></div>`;
  }).join("")||`<div class="empty">No rows match.</div>`;
  document.querySelectorAll(".item").forEach(node=>node.onclick=()=>render(Number(node.dataset.index)));
}
function init(){
  $("rowCount").textContent=SUMMARY.rows;$("uniqueCount").textContent=SUMMARY.semantic_unique;$("eventCount").textContent=SUMMARY.events;
  $("pass1").textContent=SUMMARY.pass_summary?`${(100*SUMMARY.pass_summary.pass_at_1_all).toFixed(2)}%`:"n/a";
  $("pass16").textContent=SUMMARY.pass_summary?`${(100*SUMMARY.pass_summary.pass_at_16_all).toFixed(2)}%`:"n/a";
  const cats=Object.keys(SUMMARY.categories);$("category").innerHTML=`<option value="">All categories</option>`+
    cats.map(x=>`<option value="${esc(x)}">${esc(x)} (${SUMMARY.categories[x]})</option>`).join("");
  ["search","category","filter"].forEach(id=>$(id).addEventListener(id==="search"?"input":"change",applyFilters));
  $("jumpButton").onclick=()=>{const n=Number($("jump").value);if(Number.isInteger(n)&&n>=0&&n<DATA.length)render(n)};
  $("randomButton").onclick=()=>{if(filtered.length)render(filtered[Math.floor(Math.random()*filtered.length)])};
  const match=location.hash.match(/row-(\d+)/);selected=match?Math.min(Number(match[1]),DATA.length-1):SUMMARY.curated_indices[0]||0;
  applyFilters();render(selected);
}
init();
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--pass-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="APIGen trajectory review")
    args = parser.parse_args()

    rows = read_jsonl(args.jsonl)
    events = read_jsonl(args.events) if args.events else []
    payload, summary = build_payload(rows, events)
    if args.pass_summary:
        summary["pass_summary"] = json.loads(
            args.pass_summary.read_text(encoding="utf-8")
        )
    static_list, static_content = static_review_html(payload, summary)
    output = (
        HTML_TEMPLATE.replace("__TITLE_TEXT__", html.escape(args.title))
        .replace("__STATIC_LIST_HTML__", static_list)
        .replace("__STATIC_CONTENT_HTML__", static_content)
        .replace("__PAYLOAD_JSON__", json_for_script(payload))
        .replace("__SUMMARY_JSON__", json_for_script(summary))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"HTML: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
