#!/usr/bin/env python3
"""LLM-driven generator for ToolLens tool implementations.

Reads tool definitions from toollens_tools_enhanced.jsonl, groups by category,
then prompts an LLM to generate:

1. Stateful class modules (tools/toollens/{class_key}.py) with one method per tool
2. Pydantic input schemas (tools/toollens/schemas.py)
3. Initial configs (src/toollens_config_pool.py) for diverse trajectory seeding
4. Unit tests (tests/tools/toollens/test_{class_key}.py) - per-method isolation
5. Sequential tests (tests/tools/toollens/test_sequential_{class_key}.py) - typical trajectories

Usage:
    python scripts/generate_toollens_implementations.py
    python scripts/generate_toollens_implementations.py --categories Finance,Weather,Music
    python scripts/generate_toollens_implementations.py --skip-existing --verbose
    python scripts/generate_toollens_implementations.py --only-configs   # initial configs only
    python scripts/generate_toollens_implementations.py --skip-tests     # skip test gen
"""

import argparse
import copy
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_EXTRACTION_DIR = PROJECT_ROOT / "toollens_data"
OUTPUT_DIR = PROJECT_ROOT / "tools" / "toollens"
TEST_DIR = PROJECT_ROOT / "tests" / "tools" / "toollens"
CONFIG_POOL_PATH = PROJECT_ROOT / "src" / "toollens_config_pool.py"


# ─── Configuration ──────────────────────────────────────────────────────────


def category_to_class_key(category: str) -> str:
    """Convert a ToolLens category to a snake_case class_key."""
    out = re.sub(r"[^A-Za-z0-9]+", "_", category.strip())
    out = re.sub(r"_+", "_", out).strip("_").lower()
    return out


def class_key_to_class_name(class_key: str) -> str:
    """Convert snake_case class_key to PascalCase ClassName."""
    parts = class_key.split("_")
    # Avoid trailing 'tools' suffix collision - we'll add it ourselves
    return "".join(p.capitalize() for p in parts if p)


def class_key_to_class_with_tools_suffix(class_key: str) -> str:
    """ClassName + 'Tools' suffix so it's clear these are tool collections."""
    name = class_key_to_class_name(class_key)
    if not name.endswith("Tools"):
        name = name + "Tools"
    return name


def sanitize_method_name(name: str, used: set) -> str:
    """Sanitize an api_name into a valid Python identifier.

    - Replace `/`, `-`, `.`, `{`, `}`, `[`, `]`, spaces and other non-word chars with `_`
    - Collapse repeated `_` and strip leading/trailing
    - If name doesn't start with letter/_, prepend `m_`
    - If name conflicts, append `_2`, `_3`, ...
    """
    out = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
    out = re.sub(r"_+", "_", out).strip("_")
    if not out:
        out = "method"
    if not out[0].isalpha() and out[0] != "_":
        out = "m_" + out
    base = out
    i = 2
    while out in used:
        out = f"{base}_{i}"
        i += 1
    used.add(out)
    return out


def build_method_name_map(tools: List[Dict]) -> Dict[str, str]:
    """Map each tool's api_name to a sanitized Python method name (unique within set)."""
    used = set()
    out = {}
    for t in sorted(tools, key=lambda x: x["api_name"]):
        out[t["api_name"]] = sanitize_method_name(t["api_name"], used)
    return out


# ─── Data Loading ───────────────────────────────────────────────────────────


def load_toollens_tools() -> List[Dict[str, Any]]:
    """Load all tools from toollens_tools_enhanced.jsonl."""
    path = TOOLS_EXTRACTION_DIR / "toollens_tools_enhanced.jsonl"
    if not path.exists():
        # Fall back to non-enhanced
        path = TOOLS_EXTRACTION_DIR / "toollens_tools.jsonl"
    tools = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tools.append(json.loads(line))
    return tools


def group_tools_by_category(tools: List[Dict]) -> Dict[str, List[Dict]]:
    """Group by 'category' field."""
    groups = defaultdict(list)
    for t in tools:
        groups[t["category"]].append(t)
    return dict(groups)


# ─── SimpleLLMClient (reused from BFCL generator) ───────────────────────────


class SimpleLLMClient:
    """Lightweight OpenAI-compatible LLM client (no transformers dependency)."""

    def __init__(self, url: str, api_key: str, api_model: str,
                 debug_mode: bool = False):
        self.url = url
        self.api_model = api_model
        self.debug_mode = debug_mode
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.total_calls = 0

    def chat(self, messages: list, kwargs: dict,
             max_retries: int = 10, base_delay: float = 2.0) -> tuple:
        request_timeout = kwargs.pop("timeout", 600)
        payload = {"model": self.api_model, "messages": messages, **kwargs}
        attempt = 0
        rate_limit_retries = 0

        while attempt < max_retries:
            try:
                resp = requests.request(
                    "POST",
                    url=f"{self.url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=request_timeout,
                )
                if resp.status_code >= 500:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"  [LLM] Server {resp.status_code} (attempt {attempt+1}/{max_retries}), retrying in {delay}s...")
                        time.sleep(delay)
                        attempt += 1
                        continue
                    raise RuntimeError(f"API server {resp.status_code}: {resp.text[:300]}")
                try:
                    data = resp.json()
                except Exception:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"  [LLM] JSON decode err (attempt {attempt+1}), retrying in {delay}s...")
                        time.sleep(delay)
                        attempt += 1
                        continue
                    raise RuntimeError(f"non-JSON: {resp.text[:300]}")

                if "choices" not in data:
                    if resp.status_code == 429:
                        rate_limit_retries += 1
                        delay = min(base_delay * (2 ** min(rate_limit_retries, 10)), 300)
                        print(f"  [LLM] 429, retrying in {delay}s... (#{rate_limit_retries})")
                        time.sleep(delay)
                        continue
                    raise RuntimeError(f"Unexpected ({resp.status_code}): {json.dumps(data)[:300]}")

                content = data["choices"][0]["message"]["content"] or ""

                # Strip thinking/reasoning tags (various model formats)
                reasoning = ""
                clean = content
                for tag_pair in [["\u0001", "\u0002"], ["<thinking>", "</thinking>"]]:
                    ts, te = tag_pair
                    if not ts or not te:
                        continue
                    pat = re.escape(ts) + r"(.*?)" + re.escape(te)
                    m = re.search(pat, clean, re.DOTALL)
                    if m:
                        reasoning = m.group(1).strip()
                        clean = re.sub(pat, "", clean, flags=re.DOTALL).strip()
                        break
                self.total_calls += 1
                return clean, reasoning
                self.total_calls += 1
                return clean.strip(), reasoning
            except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout):
                delay = base_delay * (2 ** attempt)
                print(f"  [LLM] Timeout (attempt {attempt+1}/{max_retries}), retrying in {delay}s...")
                time.sleep(delay)
                attempt += 1
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.HTTPError) as e:
                delay = base_delay * (2 ** attempt)
                print(f"  [LLM] Conn err (attempt {attempt+1}/{max_retries}): {e}, retry in {delay}s...")
                time.sleep(delay)
                attempt += 1

        raise RuntimeError(f"LLM call failed after {max_retries} attempts")


def create_llm_client(model: str, api_base: str, api_key: str, verbose: bool):
    url = api_base or os.getenv("OPENAI_API_BASE", "https://integrate.api.nvidia.com/v1")
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    return SimpleLLMClient(url=url, api_key=key, api_model=model, debug_mode=verbose)


def call_llm(client, system_prompt: str, user_prompt: str,
             verbose: bool = False, max_tokens: int = 16384) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs = {"temperature": 0.3, "max_tokens": max_tokens}
    print(f"  [LLM] Calling {client.api_model} (prompt={len(user_prompt)} chars)...", flush=True)
    t0 = time.time()
    response, _ = client.chat(messages=messages, kwargs=kwargs)
    elapsed = time.time() - t0
    print(f"  [LLM] Response in {elapsed:.1f}s ({len(response)} chars)", flush=True)
    return response


def extract_code_block(text: str) -> str:
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()


# ─── Prompt Builders ────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are a Python code generator producing production-quality, working Python code.

Rules:
- Return ONLY valid Python code in a single markdown code block (```python ... ```)
- Use type hints on all function signatures
- Methods return dicts matching the specified output schema EXACTLY
- Stateful methods must mutate self state appropriately
- Handle edge cases: missing args, invalid values, not-found scenarios
- Never raise exceptions from methods - return error info in the response dict
- Follow the exact parameter names from the tool definitions (keep camelCase as-is)
- Import only: json, math, re, copy, datetime, random, typing (List, Dict, Any, Optional, Tuple, Union)
- Do NOT include any explanatory text outside the code block
- Each method must have a docstring
- The class __init__ must accept initial_config: dict = None and set up internal state
- Store config values in self._config_data dict to avoid shadowing method names
- For classes with state, normalize all known config variants in __init__
- For stateless classes, methods should accept all data via parameters
- Return deterministic, sensible values that exercise realistic API behavior
- For list returns, populate with 1-3 example items
- For dict/object returns, include all schema-declared keys with realistic values"""


def build_class_prompt(class_key: str, category: str, tools: List[Dict]) -> str:
    """Build the LLM prompt for generating a complete class for one ToolLens category."""
    class_name = class_key_to_class_with_tools_suffix(class_key)
    name_map = build_method_name_map(tools)
    parts = []
    parts.append(f"## Task: Generate the {class_name} class")
    parts.append(f"File: tools/toollens/{class_key}.py")
    parts.append(f"Category: {category}")
    parts.append("")

    # Emit METHOD_NAME_MAP so ToolManager can resolve api_name → method
    parts.append("### Method name mapping (REQUIRED class attribute)")
    parts.append("Many ToolLens api_names use slashes, dots, or template syntax that aren't valid")
    parts.append("Python method names. The class MUST expose a class attribute:")
    parts.append("```python")
    parts.append("METHOD_NAME_MAP = {")
    for orig, sane in sorted(name_map.items()):
        parts.append(f'    {orig!r}: {sane!r},')
    parts.append("}")
    parts.append("```")
    parts.append("Each key is the original api_name string. Each value is the Python method name.")
    parts.append("ToolManager will use this dict to dispatch calls from the api_name.")
    parts.append("")

    # State analysis - infer from tools
    has_state = any(t.get("api_description", "").lower().startswith(("get ", "list ", "create ", "add ", "delete ", "update ", "remove ", "send ", "post ")) for t in tools)

    parts.append("### Class state analysis")
    parts.append("Analyze the API methods to determine the appropriate state to maintain:")
    parts.append("- Methods named 'get_*'/'list_*' typically read from state collections")
    parts.append("- Methods named 'create_*'/'add_*'/'post_*' typically write to state collections")
    parts.append("- Methods named 'validate_*'/'search_*' typically filter state or external data")
    parts.append("- If all methods are read-only stateless queries (e.g. external API proxies),")
    parts.append("  initialize minimal state like a counter or cache")
    parts.append("- If methods modify resources (e.g. CRUD), maintain collections keyed by ID")
    parts.append("")
    parts.append("__init__(self, initial_config: dict = None) must:")
    parts.append("- Call self._init_state() if no config given, else initialize from config")
    parts.append("- Store all keyword state fields in self._config_data dict (NOT via setattr or self.<field>)")
    parts.append("  to avoid shadowing method names with config values")
    parts.append("")

    # Tools to implement
    parts.append(f"### Tools to implement ({len(tools)} methods)")
    parts.append("")
    parts.append("Use the EXACT sanitized method names below (right column). Do NOT use the")
    parts.append("original api_name strings as method names — they are illegal Python identifiers.")
    parts.append("")

    for tool in sorted(tools, key=lambda t: t["api_name"]):
        api_name = tool["api_name"]
        method_name = name_map[api_name]
        if method_name != api_name:
            parts.append(f"#### Method: `{method_name}`  (api_name: `{api_name}`)")
        else:
            parts.append(f"#### Method: `{method_name}`")

        desc = tool.get("api_description", "")
        if desc:
            parts.append(f"Description: {desc}")

        # Parameters
        params = tool.get("parameters", {})
        props = params.get("properties", {}) if params else {}
        required = params.get("required", []) if params else []
        if props:
            parts.append("Parameters:")
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                pdefault = pinfo.get("default", None)
                is_req = pname in required
                req_str = "required" if is_req else "optional"
                default_str = ""
                if not is_req:
                    if pdefault is not None:
                        default_str = f", default={repr(pdefault)}"
                    else:
                        default_str = ", default=None"
                parts.append(f"  - {pname}: {ptype} ({req_str}{default_str}) - {pdesc}")
        else:
            parts.append("Parameters: none")

        # Output schema (replaces BFCL func_doc.response)
        output_schema = tool.get("output_schema", {})
        output_type = tool.get("output_type", "dict")
        output_desc = tool.get("output_description", "")

        if output_schema and output_schema.get("properties"):
            parts.append("Return schema (return dict must include these keys):")
            parts.append("```json")
            schema_str = json.dumps(output_schema, indent=2)
            if len(schema_str) > 1500:
                schema_str = schema_str[:1500] + "\n... (truncated)"
            parts.append(schema_str)
            parts.append("```")
        else:
            parts.append(f"Return type: {output_type}")
            if output_desc:
                parts.append(f"Return description: {output_desc[:400]}")

        # Example queries (provide context for behavior)
        eq = tool.get("example_queries", [])[:2]
        if eq:
            parts.append("Example user queries:")
            for q in eq:
                parts.append(f"  - {q[:150]}")

        parts.append("")

    parts.append("### Output format:")
    parts.append("Return the complete Python class in a single ```python ... ``` code block.")
    parts.append("The class must include __init__(self, initial_config: dict = None), the METHOD_NAME_MAP")
    parts.append("class attribute, and ALL listed methods (use the sanitized names).")
    parts.append("Each method signature must use the exact parameter names from the tool definitions.")
    parts.append("Methods must return realistic, deterministic values matching the schema.")
    parts.append("")
    return "\n".join(parts)


def build_schemas_prompt(class_key: str, tools: List[Dict]) -> str:
    """Build the LLM prompt for generating Pydantic input schemas for one category."""
    class_name = class_key_to_class_with_tools_suffix(class_key)
    name_map = build_method_name_map(tools)
    parts = []
    parts.append(f"## Task: Generate Pydantic input schemas for {class_name} tools")
    parts.append("")
    parts.append("Generate one Pydantic BaseModel per tool method, named as {SanitizedMethodName}_Input.")
    parts.append("Use the sanitized method names (right column) - NOT the original api_names.")
    parts.append("For example, if sanitized method is 'get_stock_info', schema class is 'GetStockInfoInput'.")
    parts.append("")
    parts.append("### Rules:")
    parts.append("- Use pydantic BaseModel with Field(..., description=...) for each param")
    parts.append("- Use Optional[T] = None for optional parameters")
    parts.append("- Use Literal['value1', 'value2'] for enum-typed parameters")
    parts.append("- Use List[T], Dict[str, Any], etc. for arrays/objects")
    parts.append("- Add a model_config with extra='forbid' if you want strict validation")
    parts.append("")

    parts.append("### Tools (api_name → sanitized method name → schema class):")
    for tool in sorted(tools, key=lambda t: t["api_name"]):
        api_name = tool["api_name"]
        method_name = name_map[api_name]
        schema_cls = "".join(p.capitalize() for p in method_name.split("_")) + "Input"
        params = tool.get("parameters", {}) or {}
        props = params.get("properties", {}) if params else {}
        required = params.get("required", []) if params else []
        parts.append(f"\n#### `{method_name}` (api `{api_name}`) → {schema_cls}")
        if not props:
            parts.append("(no parameters)")
            continue
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any")
            pdesc = pinfo.get("description", "")
            pdefault = pinfo.get("default", None)
            enum = pinfo.get("enum", [])
            is_req = pname in required
            extra = ""
            if enum:
                extra = f" [enum: {enum}]"
            parts.append(f"  - {pname}: {ptype} ({'required' if is_req else 'optional'}){extra} - {pdesc}")

    parts.append("")
    parts.append("Return all schema definitions in a single ```python ... ``` code block.")
    return "\n".join(parts)


def build_tests_prompt(class_key: str, class_code: str,
                       canonical_config: Dict, tools: List[Dict]) -> str:
    """Build the LLM prompt for generating unit tests (in isolation)."""
    class_name = class_key_to_class_with_tools_suffix(class_key)
    name_map = build_method_name_map(tools)
    parts = []
    parts.append(f"## Task: Generate pytest unit tests for {class_name}")
    parts.append("")
    parts.append(f"File: tests/tools/toollens/test_{class_key}.py")
    parts.append("")
    parts.append("### Requirements:")
    parts.append(f"- Generate 1-2 tests per method ({len(tools)} methods - use sanitized names)")
    parts.append("- Use pytest fixtures for class instance setup")
    parts.append("- Test normal operation, edge cases (params=None, empty inputs), and error handling")
    parts.append("- Test that return type matches expected (dict for object, list for list, etc.)")
    parts.append(f"- Import the class from tools.toollens.{class_key}")
    parts.append("- Each test should call ONE method in isolation (NOT sequences)")
    parts.append("- Do NOT use unittest.TestCase - use pytest functions + fixtures")
    parts.append("- Use the SANITIZED method names (right column) when calling methods")
    parts.append("")
    parts.append("### Test structure:")
    parts.append("```python")
    parts.append("import pytest, json")
    parts.append(f"from tools.toollens.{class_key} import {class_name}")
    parts.append("")
    parts.append("@pytest.fixture")
    parts.append(f"def {class_key}_instance():")
    if canonical_config:
        parts.append(f"    config = {repr(canonical_config)}")
    else:
        parts.append(f"    config = None  # stateless")
    parts.append(f"    return {class_name}(initial_config=config)")
    parts.append("```")
    parts.append("")

    if canonical_config:
        parts.append("### initial_config for fixtures:")
        parts.append("```json")
        parts.append(json.dumps(canonical_config, indent=2)[:3000])
        parts.append("```")
        parts.append("")
    else:
        parts.append("### Note: This class is stateless - pass initial_config=None")
        parts.append("")

    parts.append("### Methods (call each in isolation, use sanitized name):")
    for tool in sorted(tools, key=lambda t: t["api_name"]):
        api_name = tool["api_name"]
        method_name = name_map[api_name]
        if method_name != api_name:
            marker = f"`{method_name}` (api_name: `{api_name}`)"
        else:
            marker = f"`{method_name}`"
        params = tool.get("parameters", {}) or {}
        props = params.get("properties", {})
        param_names = list(props.keys())
        parts.append(f"- {marker}({', '.join(param_names) if param_names else ''})")
    parts.append("")

    # Include class code (truncated for reference)
    parts.append("### Generated class code (for reference):")
    parts.append("```python")
    if len(class_code) > 4000:
        lines = class_code.split("\n")
        kept = []
        for line in lines:
            if (line.strip().startswith(("def ", "class ", "@", '"""', "'''"))
                    or not line.strip()):
                kept.append(line)
            elif len(kept) > 0 and not line.startswith(" "):
                kept.append(line)
        truncated = "\n".join(kept)
        if len(truncated) > 4000:
            truncated = truncated[:4000] + "\n# ... (truncated)"
        parts.append(truncated)
    else:
        parts.append(class_code)
    parts.append("```")
    parts.append("")

    parts.append("Return the complete test file in a single ```python ... ``` code block.")
    return "\n".join(parts)


def build_sequential_tests_prompt(class_key: str, class_code: str,
                                  canonical_config: Dict,
                                  tools: List[Dict]) -> str:
    """Build the LLM prompt for generating sequential stateful tests."""
    class_name = class_key_to_class_with_tools_suffix(class_key)
    name_map = build_method_name_map(tools)
    parts = []
    parts.append(f"## Task: Generate sequential API tests for {class_name}")
    parts.append("")
    parts.append(f"File: tests/tools/toollens/test_sequential_{class_key}.py")
    parts.append("")
    parts.append("### Overview:")
    parts.append("Generate tests that exercise multi-call sequences representing typical")
    parts.append("user trajectories through this API collection.")
    parts.append("")
    parts.append("### Test classes to generate:")
    parts.append("1. `Test{class_name}SequentialCorrect` - Correct ordered sequences")
    parts.append("   - call method A → call method B that depends on or builds on A")
    parts.append("   - for stateful APIs: create resource → get resource → verify")
    parts.append("   - for stateless APIs: combine search/filter queries in sequence")
    parts.append("")
    parts.append("2. `Test{class_name}SequentialProblematic` - Problematic sequences")
    parts.append("   - get nonexistent resource → verify empty/error response")
    parts.append("   - call with invalid params → next method should not crash")
    parts.append("   - call methods in wrong order if ordering matters")
    parts.append("")
    parts.append("### Requirements:")
    parts.append("- Use pytest fixtures (NOT unittest.TestCase)")
    parts.append("- Each test must call 2+ methods in sequence (use sanitized names)")
    parts.append("- Use json.loads(json.dumps(config)) for deep copy in fixtures")
    parts.append("- Import: import pytest, import json")
    parts.append(f"- Import: from tools.toollens.{class_key} import {class_name}")
    parts.append("- Start with fresh instance per test (no shared state across tests)")
    parts.append("")

    if canonical_config:
        parts.append("### initial_config for fixtures:")
        parts.append("```json")
        parts.append(json.dumps(canonical_config, indent=2)[:3000])
        parts.append("```")
    else:
        parts.append("### initial_config: None (stateless)")
    parts.append("")

    # Class code reference
    parts.append("### Class code (for reference):")
    parts.append("```python")
    if len(class_code) > 4000:
        lines = class_code.split("\n")
        kept = [l for l in lines if l.strip().startswith(("def ", "class ", "@", '"""', "'''"))
                or not l.strip()]
        truncated = "\n".join(kept)[:4000]
        parts.append(truncated + "\n# ...(truncated)")
    else:
        parts.append(class_code)
    parts.append("```")
    parts.append("")

    parts.append("### Tools available (use sanitized method names):")
    for tool in sorted(tools, key=lambda t: t["api_name"]):
        name = name_map[tool["api_name"]]
        parts.append(f"- `{name}`")
    parts.append("")

    parts.append("Return the complete test file in a single ```python ... ``` code block.")
    parts.append("Generate 3-5 correct sequence tests and 3-5 problematic sequence tests.")
    parts.append("")
    return "\n".join(parts)


def build_initial_config_prompt(class_key: str, category: str,
                                tools: List[Dict]) -> str:
    """Build the LLM prompt to infer an initial_config for a stateful class."""
    class_name = class_key_to_class_with_tools_suffix(class_key)
    name_map = build_method_name_map(tools)
    parts = []
    parts.append(f"## Task: Design initial_config for {class_name}")
    parts.append("")
    parts.append("Generate a JSON initial_config dict that exercises the most diverse")
    parts.append(f"set of trajectories across the {len(tools)} methods in the {category} category.")
    parts.append("")
    parts.append("### Rules:")
    parts.append("- Output ONE JSON object (no surrounding text, no markdown)")
    parts.append("- Provide 3-5 sample collections/records that exercise read methods")
    parts.append("- Provide known IDs/keys that 'get_*' methods would look up")
    parts.append("- Provide some empty/sparse collections to test edge cases")
    parts.append("- Use realistic values (e.g., real city names, real product names)")
    parts.append("- Keep total size < 2KB")
    parts.append("- If the class is stateless, return {} (empty dict)")
    parts.append("")
    parts.append("### Methods to support (sanitized_name | api_name | description):")
    for tool in sorted(tools, key=lambda t: t["api_name"])[:30]:
        api_name = tool["api_name"]
        method_name = name_map[api_name]
        desc = (tool.get("api_description") or "")[:80]
        params = (tool.get("parameters") or {}).get("properties", {})
        params_str = ", ".join(f"{k}:{v.get('type', '?')}" for k, v in params.items())
        parts.append(f"- `{method_name}` | api `{api_name}` | {desc}")
    if len(tools) > 30:
        parts.append(f"... and {len(tools) - 30} more")
    parts.append("")

    parts.append("Return ONLY a minified JSON object (no comments, no markdown fences).")
    return "\n".join(parts)


# ─── Validation ─────────────────────────────────────────────────────────────


def validate_python_code(code: str, filename: str) -> Tuple[bool, str]:
    try:
        compile(code, filename, "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def run_tests(test_dir: Path, class_key: str, verbose: bool = False) -> Tuple[bool, str]:
    """Run pytest on the generated test file (or all if class_key='ALL')."""
    if class_key == "ALL":
        test_path = str(test_dir)
    else:
        test_file = test_dir / f"test_{class_key}.py"
        if not test_file.exists():
            return False, f"Test file not found: {test_file}"
        test_path = str(test_file)

    cmd = ["python3", "-m", "pytest", test_path, "-v", "--tb=short", "-x"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                            cwd=str(PROJECT_ROOT))
    output = result.stdout + result.stderr
    return result.returncode == 0, output


# ─── File Writing ───────────────────────────────────────────────────────────


def write_class_file(output_dir: Path, class_key: str, code: str,
                     class_name: str, verbose: bool = False):
    filepath = output_dir / f"{class_key}.py"
    header = f'"""Auto-generated {class_name} implementation."""\n\n'
    if not (code.startswith('"""') or code.startswith("'''")):
        code = header + code
    filepath.write_text(code)
    if verbose:
        print(f"  Wrote {filepath} ({len(code)} bytes)")


def write_schemas_file(output_dir: Path, all_schemas: Dict[str, str], verbose: bool = False):
    filepath = output_dir / "schemas.py"
    parts = ['"""Auto-generated Pydantic input schemas for all ToolLens tools."""\n\n']
    parts.append("from pydantic import BaseModel, Field\n")
    parts.append("from typing import Optional, List, Dict, Any, Literal, Union\n\n")
    for class_key in sorted(all_schemas.keys()):
        class_name = class_key_to_class_with_tools_suffix(class_key)
        parts.append(f"# ─── {class_name} ──────────\n\n")
        lines = all_schemas[class_key].split("\n")
        filtered = [l for l in lines
                    if not l.strip().startswith(("from pydantic", "import pydantic",
                                                  "from typing", "import typing"))]
        parts.append("\n".join(filtered))
        parts.append("\n\n")
    filepath.write_text("".join(parts))
    if verbose:
        print(f"  Wrote {filepath}")


def write_test_file(test_dir: Path, class_key: str, code: str,
                    prefix: str = "test_", verbose: bool = False):
    filepath = test_dir / f"{prefix}{class_key}.py"
    filepath.write_text(code)
    if verbose:
        print(f"  Wrote {filepath} ({len(code)} bytes)")


def write_conftest(test_dir: Path, verbose: bool = False):
    filepath = test_dir / "conftest.py"
    if filepath.exists():
        return
    code = '"""Shared fixtures for ToolLens tests."""\n\nimport pytest\nimport json\n'
    filepath.write_text(code)
    if verbose:
        print(f"  Wrote {filepath}")


def write_init_file(output_dir: Path, class_keys: List[str], verbose: bool = False):
    """Write tools/toollens/__init__.py with the class registry."""
    filepath = output_dir / "__init__.py"
    parts = ['"""ToolLens tool class registry (auto-generated)."""\n\n']
    parts.append("from typing import Dict, Any, Optional\n\n")
    parts.append("TOOLLENS_TOOL_CLASSES: Dict[str, str] = {\n")
    for ck in sorted(class_keys):
        cn = class_key_to_class_with_tools_suffix(ck)
        parts.append(f'    "{ck}": "tools.toollens.{ck}",\n')
    parts.append("}\n\n")
    parts.append("TOOLLENS_CLASS_NAMES: Dict[str, str] = {\n")
    for ck in sorted(class_keys):
        cn = class_key_to_class_with_tools_suffix(ck)
        parts.append(f'    "{ck}": "{cn}",\n')
    parts.append("}\n\n")
    # Eagerly build api_name -> class_key from each class's METHOD_NAME_MAP
    parts.append("TOOLLENS_API_NAME_TO_CLASS_KEY: Dict[str, str] = {}\n")
    parts.append("_APIS_POPULATED = False\n")
    parts.append("def _populate_api_name_map() -> None:\n")
    parts.append("    global _APIS_POPULATED\n")
    parts.append("    if _APIS_POPULATED:\n")
    parts.append("        return\n")
    parts.append("    import importlib, inspect\n")
    parts.append("    for class_key, module_path in TOOLLENS_TOOL_CLASSES.items():\n")
    parts.append("        try:\n")
    parts.append("            mod = importlib.import_module(module_path)\n")
    parts.append("            cls = getattr(mod, TOOLLENS_CLASS_NAMES[class_key], None)\n")
    parts.append("            if cls is None:\n")
    parts.append("                continue\n")
    parts.append("            if hasattr(cls, 'METHOD_NAME_MAP'):\n")
    parts.append("                for api_name in cls.METHOD_NAME_MAP:\n")
    parts.append("                    TOOLLENS_API_NAME_TO_CLASS_KEY[api_name] = class_key\n")
    parts.append("            else:\n")
    parts.append("                for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction):\n")
    parts.append("                    if name.startswith('_'):\n")
    parts.append("                        continue\n")
    parts.append("                    TOOLLENS_API_NAME_TO_CLASS_KEY[name] = class_key\n")
    parts.append("        except Exception as e:\n")
    parts.append("            print(f'ToolLens: Could not introspect class {class_key}: {e}')\n")
    parts.append("    _APIS_POPULATED = True\n\n")
    parts.append("def reset_api_name_map() -> None:\n")
    parts.append("    global _APIS_POPULATED\n")
    parts.append("    TOOLLENS_API_NAME_TO_CLASS_KEY.clear()\n")
    parts.append("    _APIS_POPULATED = False\n\n")
    parts.append("def get_toollens_api_name_map() -> Dict[str, str]:\n")
    parts.append("    _populate_api_name_map()\n")
    parts.append("    return dict(TOOLLENS_API_NAME_TO_CLASS_KEY)\n\n")
    parts.append("def create_toollens_instance(class_key: str, initial_config: dict = None):\n")
    parts.append("    import importlib\n")
    parts.append("    if class_key not in TOOLLENS_TOOL_CLASSES:\n")
    parts.append("        raise KeyError(f'Unknown ToolLens class key: {class_key}')\n")
    parts.append("    module = importlib.import_module(TOOLLENS_TOOL_CLASSES[class_key])\n")
    parts.append("    cls = getattr(module, TOOLLENS_CLASS_NAMES[class_key])\n")
    parts.append("    return cls(initial_config=initial_config)\n\n")
    parts.append("def create_toollens_tool_instances(\n")
    parts.append("    configs: Optional[Dict[str, Dict[str, Any]]] = None\n")
    parts.append(") -> Dict[str, Any]:\n")
    parts.append("    '''Instantiate every ToolLens class with its per-class config.\n")
    parts.append("\n")
    parts.append("    Args:\n")
    parts.append("        configs: Dict mapping class_key -> config dict. If None or\n")
    parts.append("                missing a key, the class is instantiated with initial_config=None.\n")
    parts.append("\n")
    parts.append("    Returns:\n")
    parts.append("        Dict mapping class_key -> instance.\n")
    parts.append("    '''\n")
    parts.append("    configs = configs or {}\n")
    parts.append("    instances: Dict[str, Any] = {}\n")
    parts.append("    for class_key in TOOLLENS_TOOL_CLASSES:\n")
    parts.append("        try:\n")
    parts.append("            cfg = configs.get(class_key, {}) or None\n")
    parts.append("            instances[class_key] = create_toollens_instance(class_key, initial_config=cfg)\n")
    parts.append("        except Exception as e:\n")
    parts.append("            print(f'Warning: Could not instantiate ToolLens class {class_key}: {e}')\n")
    parts.append("    return instances\n")
    parts.append("\n__all__ = [\n")
    parts.append("    'TOOLLENS_TOOL_CLASSES', 'TOOLLENS_CLASS_NAMES',\n")
    parts.append("    'TOOLLENS_API_NAME_TO_CLASS_KEY',\n")
    parts.append("    'create_toollens_instance', 'create_toollens_tool_instances',\n")
    parts.append("    'get_toollens_api_name_map', 'reset_api_name_map',\n")
    parts.append("]\n")
    filepath.write_text("".join(parts))
    if verbose:
        print(f"  Wrote {filepath}")


def append_to_config_pool(class_key: str, class_name: str, config: Dict,
                          verbose: bool = False):
    """Append a generated initial_config to src/toollens_config_pool.py."""
    if not CONFIG_POOL_PATH.exists():
        CONFIG_POOL_PATH.write_text(
            '"""ToolLens initial config pool (auto-generated).\n\n'
            'Provides per-class config variations for diverse trajectory generation.\n"""\n\n'
            'from typing import Dict, Any\n\n'
            'TOOLLENS_CONFIGS: Dict[str, list] = {}\n\n'
            'def get_toollens_configs(class_key: str) -> list:\n'
            '    return TOOLLENS_CONFIGS.get(class_key, [{}])\n\n'
        )
    with open(CONFIG_POOL_PATH, "a") as f:
        f.write(f"\n# ─── {class_name} ({class_key}) ──────────\n")
        f.write(f'TOOLLENS_CONFIGS["{class_key}"] = [\n')
        f.write(f"    {repr(config)},\n")
        f.write("]\n")
    if verbose:
        print(f"  Appended config for {class_key} to {CONFIG_POOL_PATH.name}")


# ─── Generation Orchestration ───────────────────────────────────────────────


def generate_class(
    class_key: str,
    category: str,
    client,
    tools: List[Dict],
    output_dir: Path,
    test_dir: Path,
    skip_existing: bool,
    verbose: bool,
    max_retries: int,
    skip_tests: bool,
    skip_configs: bool,
    skip_sequential: bool,
) -> Tuple[bool, str, str, Optional[Dict]]:
    """Generate class + schemas + tests + initial_config for one category."""
    class_name = class_key_to_class_with_tools_suffix(class_key)
    class_file = output_dir / f"{class_key}.py"

    if skip_existing and class_file.exists():
        print(f"  [SKIP] {class_name} - exists")
        return True, class_file.read_text(), "", None

    print(f"\n{'='*60}")
    print(f"Generating: {class_name} ({len(tools)} tools, category={category})")
    print(f"{'='*60}")

    # ── Step 1: Generate class code ──
    print(f"  [1/5] Generating class code...")
    prompt = build_class_prompt(class_key, category, tools)
    name_map = build_method_name_map(tools)
    expected_method_defs = {f"def {sane}" for sane in name_map.values()}
    class_code = ""
    missing: list = []
    for attempt in range(max_retries + 1):
        response = call_llm(client, SYSTEM_PROMPT, prompt, verbose=verbose)
        class_code = extract_code_block(response)
        valid, error = validate_python_code(class_code, f"{class_key}.py")
        if not valid:
            print(f"  [1/5] ✗ Syntax: {error}")
            if attempt < max_retries:
                prompt += f"\n\n### PREVIOUS HAD SYNTAX ERROR:\n{error}\nFix and regenerate."
            continue
        # Verify all sanitized method names are present
        missing = []
        found_names = []
        for orig, sane in name_map.items():
            if f"def {sane}" in class_code:
                found_names.append(sane)
            else:
                missing.append(sane)
        if missing:
            # Diagnostic: show all `def X(` lines the LLM actually wrote
            actual_defs = re.findall(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", class_code)
            user_methods = [d for d in actual_defs if d not in ("__init__", "_init_state")]
            print(f"  [1/5] ⚠ Missing: {missing[:5]}{'...' if len(missing)>5 else ''}")
            print(f"    (Generated method names: {user_methods[:8]}{'...' if len(user_methods)>8 else ''})")
            print(f"    (Expected vs missing: {[(s, name_map.get(s, s)) for s in missing[:5]]})")
            if attempt < max_retries:
                # Give explicit copy-pasteable method signatures to use verbatim
                explicit = "\n".join(f"def {s}(self, ...):  # api_name: {o}"
                                    for o, s in name_map.items() if s in missing)
                prompt += f"\n\n### MISSING METHODS — generate them with EXACTLY these names:\n```python\n{explicit}\n```\nThese names are NOT case-flexible. Generate the class again with all methods present."
            continue
        print(f"  [1/5] ✓ Class validated ({len(class_code)} chars)")
        break

    if not class_code:
        print(f"  [1/5] ✗ Failed")
        return False, "", "", None

    write_class_file(output_dir, class_key, class_code, class_name, verbose)
    if missing:
        print(f"  [1/5] ⚠ Saved with {len(missing)} missing methods (best-effort)")
    time.sleep(1)

    # ── Step 2: Generate Pydantic schemas ──
    print(f"  [2/5] Generating schemas...")
    schema_prompt = build_schemas_prompt(class_key, tools)
    schemas_code = ""
    for attempt in range(max_retries + 1):
        response = call_llm(client, SYSTEM_PROMPT, schema_prompt, verbose=verbose)
        schemas_code = extract_code_block(response)
        if validate_python_code("from pydantic import BaseModel\n" + schemas_code, "schemas.py")[0]:
            print(f"  [2/5] ✓ Schemas validated")
            break
        error = validate_python_code("from pydantic import BaseModel\n" + schemas_code, "schemas.py")[1]
        print(f"  [2/5] ✗ Syntax: {error}")
        if attempt < max_retries:
            schema_prompt += f"\n\n### SYNTAX ERROR:\n{error}\nFix and regenerate."
    time.sleep(1)

    # ── Step 3: Generate initial_config ──
    print(f"  [3/5] Generating initial_config...")
    canonical_config = None
    if not skip_configs:
        config_prompt = build_initial_config_prompt(class_key, category, tools)
        config_json = ""
        for attempt in range(max_retries + 1):
            response = call_llm(client, "You respond with ONLY a minified JSON object.",
                                config_prompt, verbose=verbose, max_tokens=4000)
            text = response.strip()
            # Strip code fences
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                text = text[start:end+1]
            try:
                canonical_config = json.loads(text)
                print(f"  [3/5] ✓ Config generated ({len(json.dumps(canonical_config))} chars)")
                append_to_config_pool(class_key, class_name, canonical_config, verbose)
                break
            except json.JSONDecodeError as e:
                print(f"  [3/5] ✗ JSON parse: {e}")
                if attempt < max_retries:
                    config_prompt += f"\n\n### PREVIOUS INVALID JSON: {e}\nReturn ONLY valid JSON."
    else:
        print(f"  [3/5] Skipping initial_config generation (--skip-configs)")
    time.sleep(1)

    # ── Step 4: Generate unit tests ──
    if skip_tests:
        print(f"  [4/5] Skipping tests (--skip-tests)")
    else:
        print(f"  [4/5] Generating unit tests...")
        test_prompt = build_tests_prompt(class_key, class_code, canonical_config or {}, tools)
        tests_code = ""
        for attempt in range(max_retries + 1):
            response = call_llm(client, SYSTEM_PROMPT, test_prompt, verbose=verbose, max_tokens=8192)
            tests_code = extract_code_block(response)
            if validate_python_code(tests_code, f"test_{class_key}.py")[0]:
                print(f"  [4/5] ✓ Tests validated")
                write_test_file(test_dir, class_key, tests_code, prefix="test_", verbose=verbose)
                break
            error = validate_python_code(tests_code, f"test_{class_key}.py")[1]
            print(f"  [4/5] ✗ Syntax: {error}")
            if attempt < max_retries:
                test_prompt += f"\n\n### SYNTAX ERROR:\n{error}\nFix and regenerate."
    time.sleep(1)

    # ── Step 5: Generate sequential tests ──
    if skip_sequential or skip_tests:
        print(f"  [5/5] Skipping sequential tests")
    else:
        print(f"  [5/5] Generating sequential tests...")
        seq_prompt = build_sequential_tests_prompt(class_key, class_code, canonical_config or {}, tools)
        seq_code = ""
        for attempt in range(max_retries + 1):
            response = call_llm(client, SYSTEM_PROMPT, seq_prompt, verbose=verbose, max_tokens=8192)
            seq_code = extract_code_block(response)
            if validate_python_code(seq_code, f"test_sequential_{class_key}.py")[0]:
                print(f"  [5/5] ✓ Sequential tests validated")
                write_test_file(test_dir, class_key, seq_code, prefix="test_sequential_", verbose=verbose)
                break
            error = validate_python_code(seq_code, f"test_sequential_{class_key}.py")[1]
            print(f"  [5/5] ✗ Syntax: {error}")
            if attempt < max_retries:
                seq_prompt += f"\n\n### SYNTAX ERROR:\n{error}\nFix and regenerate."

    return True, class_code, schemas_code, canonical_config


# ─── Main ───────────────────────────────────────────────────────────────────


import subprocess  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="LLM-driven generator for ToolLens tool implementations"
    )
    parser.add_argument(
        "--categories", type=str, default=None,
        help="Comma-separated ToolLens categories to generate (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--test-dir", type=str, default=str(TEST_DIR),
        help=f"Test directory (default: {TEST_DIR})",
    )
    parser.add_argument(
        "--model", type=str, default="z-ai/glm-5.2",
        help="LLM model (default: z-ai/glm-5.2)",
    )
    parser.add_argument(
        "--api-base", type=str, default=None,
        help="API base URL (default: from OPENAI_API_BASE or NVIDIA NIM)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key (default: from OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip classes whose .py file already exists",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help="Max retries per generation step (default: 2)",
    )
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip unit + sequential test generation")
    parser.add_argument("--skip-configs", action="store_true",
                        help="Skip initial_config generation")
    parser.add_argument("--skip-sequential", action="store_true",
                        help="Skip sequential tests only (keep unit tests)")
    parser.add_argument("--only-tests", action="store_true",
                        help="Only regenerate tests for existing class files")
    parser.add_argument("--only-configs", action="store_true",
                        help="Only regenerate initial_configs for existing class files")
    parser.add_argument("--limit-tools", type=int, default=None,
                        help="Limit number of tools per category (debug). E.g. --limit-tools 10 takes first 10")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    test_dir = Path(args.test_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading ToolLens tools...")
    all_tools = load_toollens_tools()
    print(f"  Loaded {len(all_tools)} tools")
    by_cat = group_tools_by_category(all_tools)
    print(f"  {len(by_cat)} categories")

    # Filter target categories
    if args.categories:
        wanted = set(c.strip() for c in args.categories.split(","))
        by_cat = {c: ts for c, ts in by_cat.items() if c in wanted}
        if not by_cat:
            print(f"ERROR: No matching categories. Available: {sorted(by_cat)}")
            sys.exit(1)

    print(f"  Target categories: {sorted(by_cat.keys())}")
    print(f"  Model: {args.model}")

    # Build the LLM client
    print(f"\nCreating LLM client...")
    client = create_llm_client(args.model, args.api_base, args.api_key, args.verbose)

    write_conftest(test_dir, args.verbose)

    all_schemas: Dict[str, str] = {}
    results: Dict[str, bool] = {}
    generated_class_keys: List[str] = []

    for category in sorted(by_cat.keys()):
        tools = by_cat[category]
        if args.limit_tools:
            tools = tools[: args.limit_tools]
        class_key = category_to_class_key(category)
        class_name = class_key_to_class_with_tools_suffix(class_key)
        generated_class_keys.append(class_key)

        if args.only_tests:
            class_file = output_dir / f"{class_key}.py"
            if not class_file.exists():
                print(f"  [SKIP] {class_name} - no class file")
                continue
            class_code = class_file.read_text()
            test_prompt = build_tests_prompt(class_key, class_code, {}, tools)
            for attempt in range(args.max_retries + 1):
                response = call_llm(client, SYSTEM_PROMPT, test_prompt,
                                   verbose=args.verbose, max_tokens=8192)
                tests_code = extract_code_block(response)
                if validate_python_code(tests_code, f"test_{class_key}.py")[0]:
                    write_test_file(test_dir, class_key, tests_code,
                                    prefix="test_", verbose=args.verbose)
                    results[class_key] = True
                    break
                if attempt < args.max_retries:
                    test_prompt += "\n\n### ERROR\n" + validate_python_code(tests_code, "x")[1]
            continue

        if args.only_configs:
            if not args.skip_tests:
                # only-configs implies skip-tests effectively for the test phase
                pass
            config_prompt = build_initial_config_prompt(class_key, category, tools)
            response = call_llm(client, "Return ONLY a minified JSON object.",
                                config_prompt, verbose=args.verbose, max_tokens=4000)
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            s = text.find("{")
            e = text.rfind("}")
            if 0 <= s < e:
                try:
                    cfg = json.loads(text[s:e+1])
                    append_to_config_pool(class_key, class_name, cfg, args.verbose)
                    results[class_key] = True
                except Exception as ex:
                    print(f"  ✗ config parse error: {ex}")
                    results[class_key] = False
            continue

        success, class_code, schemas_code, canonical_config = generate_class(
            class_key=class_key,
            category=category,
            client=client,
            tools=tools,
            output_dir=output_dir,
            test_dir=test_dir,
            skip_existing=args.skip_existing,
            verbose=args.verbose,
            max_retries=args.max_retries,
            skip_tests=args.skip_tests,
            skip_configs=args.skip_configs,
            skip_sequential=args.skip_sequential,
        )
        results[class_key] = success
        if schemas_code:
            all_schemas[class_key] = schemas_code

    # ── Write combined schemas.py ──
    if all_schemas:
        write_schemas_file(output_dir, all_schemas, args.verbose)

    # ── Write tools/toollens/__init__.py ──
    write_init_file(output_dir, generated_class_keys, args.verbose)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Generation Summary")
    print(f"{'='*60}")
    for ck, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {class_key_to_class_with_tools_suffix(ck)} ({ck})")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  Total: {passed}/{total}")

    # ── Optionally run tests ──
    if passed == total and total > 0 and not args.skip_tests:
        print(f"\nRunning all generated tests...")
        ok, output = run_tests(test_dir, "ALL", args.verbose)
        if ok:
            print("  ✓ All tests passed!")
        else:
            print("  ✗ Some tests failed:")
            for line in output.split("\n")[-40:]:
                print(f"    {line}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
