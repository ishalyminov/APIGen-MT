"""Shared policy-visible OpenAI tool-schema projection.

Training and native BFCL evaluation must render the same function surface.  In
particular, output/response schemas are simulator metadata: the policy learns
their values from tool results, not from a schema field attached to the tool.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Iterable


TOOL_SCHEMA_PROJECTION = "openai_name_description_parameters_v3"

GORILLA_TO_OPENAI = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}


def _canonical_schema(value: Any, *, nested_item: bool = False) -> Any:
    """Apply BFCL's current Gorilla-to-OpenAI input-type conversion exactly."""
    if isinstance(value, list):
        return [_canonical_schema(item, nested_item=nested_item) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)

    result = {
        key: _canonical_schema(item, nested_item=(key == "items"))
        for key, item in value.items()
    }
    raw_type = result.get("type")
    if isinstance(raw_type, str):
        mapped = GORILLA_TO_OPENAI.get(raw_type, "string")
        result["type"] = mapped
        # BFCL's converter only annotates float-valued object properties; its
        # separate array-item path maps the type without adding format/text.
        if raw_type == "float" and not nested_item:
            result["format"] = "float"
            description = result.get("description", "")
            if not isinstance(description, str):
                description = str(description)
            suffix = " This is a float type value."
            if not description.endswith(suffix):
                result["description"] = description + suffix
    return result


def canonical_function(
    function: dict[str, Any],
    *,
    removed_input_properties: Iterable[str] = (),
) -> dict[str, Any]:
    """Return exactly the policy-visible ``name/description/parameters`` fields."""
    raw = function.get("function") if isinstance(function.get("function"), dict) else function
    name = raw.get("name")
    description = raw.get("description")
    parameters = raw.get("parameters")
    if not isinstance(name, str) or not name:
        raise ValueError("tool function has no non-empty name")
    if not isinstance(description, str):
        raise ValueError(f"tool {name!r} has no string description")
    if not isinstance(parameters, dict):
        raise ValueError(f"tool {name!r} parameters are not an object schema")

    projected = _canonical_schema(parameters)
    projected["type"] = "object"
    properties = projected.get("properties")
    if not isinstance(properties, dict):
        raise ValueError(f"tool {name!r} parameters have no properties object")

    removed = set(removed_input_properties)
    for key in removed:
        properties.pop(key, None)
    required = projected.get("required")
    if isinstance(required, list) and removed:
        projected["required"] = [key for key in required if key not in removed]

    # BFCL's native OpenAI path replaces dots before the request reaches vLLM.
    policy_name = re.sub(r"\.", "_", name)
    return {
        "name": policy_name,
        "description": description,
        "parameters": projected,
    }


def canonical_openai_tools(
    functions: Iterable[dict[str, Any]],
    *,
    removed_input_properties: dict[str, Iterable[str]] | None = None,
) -> list[dict[str, Any]]:
    removed_input_properties = removed_input_properties or {}
    tools = []
    for function in functions:
        raw = function.get("function") if isinstance(function.get("function"), dict) else function
        name = raw.get("name")
        removed = removed_input_properties.get(name, ()) if isinstance(name, str) else ()
        tools.append(
            {
                "type": "function",
                "function": canonical_function(
                    raw, removed_input_properties=removed
                ),
            }
        )
    return tools
