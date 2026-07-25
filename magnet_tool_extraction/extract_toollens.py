#!/usr/bin/env python3
"""
Extract tools from ToolLens dataset.

ToolLens (Tool-COLT/ToolLens) contains:
- corpus.jsonl: 464 tool definitions with category, name, description, parameters, return_schema
- ToolLens_data.json: 18,770 query-API pairs with template_response (return schemas)

Output format similar to bfcl_v3_tools_with_outputs.jsonl for compatibility.
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import os
from dotenv import load_dotenv

from llm_client import LocalOpenAILLMClient

# Type normalization from ToolLens informal types to JSON Schema types
TYPE_MAPPING = {
    'str': 'string',
    'bool': 'boolean',
    'int': 'integer',
    'float': 'number',
    'double': 'number',
    'list of str': 'array',
    'list of int': 'array',
    'list of float': 'array',
    'list of dict': 'array',
    'empty list': 'array',
    'object': 'dict',
}


def normalize_type(type_str: str) -> str:
    """Normalize type string to JSON Schema compatible type."""
    if not type_str:
        return 'string'

    type_str = type_str.strip().lower()

    # Check for list types
    if type_str.startswith('list of '):
        return 'array'

    # Check for dict/object
    if type_str in ('dict', 'object', '{}', 'empty object'):
        return 'dict'

    # Check for null/none
    if type_str in ('null', 'none', 'nonetype'):
        return 'null'

    return TYPE_MAPPING.get(type_str, type_str)


def parse_toollens_text(text: str) -> Optional[Dict[str, Any]]:
    """Parse the combined text field from corpus.jsonl into structured data.

    Format: category_name:X, tool_name:Y, api_name:Z, api_description:..., required_params:[...], optional_params:[...], return_schema:{...}
    """
    try:
        result = {}

        # Extract category_name
        match = re.search(r'category_name:([^,]+)', text)
        if match:
            result['category'] = match.group(1).strip()

        # Extract tool_name
        match = re.search(r'tool_name:([^,]+)', text)
        if match:
            result['tool_name'] = match.group(1).strip()

        # Extract api_name
        match = re.search(r'api_name:([^,]+)', text)
        if match:
            result['api_name'] = match.group(1).strip()

        # Extract api_description (everything up to required_params)
        match = re.search(r'api_description:(.+?)(?:,\s*required_params:|$)', text, re.DOTALL)
        if match:
            desc = match.group(1).strip()
            # Clean up markdown formatting
            desc = re.sub(r'\*\*([^*]+)\*\*', r'\1', desc)  # Remove bold
            result['api_description'] = desc

        # Extract required_params (JSON array)
        match = re.search(r'required_params:\s*(\[[^\]]*\])', text)
        if match:
            try:
                result['required_params'] = json.loads(match.group(1))
            except json.JSONDecodeError:
                result['required_params'] = []

        # Extract optional_params (JSON array)
        match = re.search(r'optional_params:\s*(\[[^\]]*\])', text)
        if match:
            try:
                result['optional_params'] = json.loads(match.group(1))
            except json.JSONDecodeError:
                result['optional_params'] = []

        # Extract return_schema (JSON object)
        match = re.search(r'return_schema:\s*(\{.*\})', text, re.DOTALL)
        if match:
            try:
                result['return_schema'] = json.loads(match.group(1))
            except json.JSONDecodeError:
                result['return_schema'] = {}

        return result

    except Exception as e:
        print(f"Error parsing text: {e}")
        return None


def build_parameters_schema(required_params: List, optional_params: List) -> Dict[str, Any]:
    """Build JSON Schema for parameters from required and optional param lists."""
    properties = {}
    required = []

    # Process required params
    for param in required_params:
        name = param.get('name', '')
        ptype = param.get('type', 'STRING')
        desc = param.get('description', '')

        if name:
            properties[name] = {
                'type': ptype.lower(),
                'description': desc
            }
            required.append(name)

    # Process optional params
    for param in optional_params:
        name = param.get('name', '')
        ptype = param.get('type', 'STRING')
        desc = param.get('description', '')
        default = param.get('default')

        if name:
            prop = {
                'type': ptype.lower(),
                'description': desc
            }
            if default is not None:
                prop['default'] = default
            properties[name] = prop

    schema = {
        'type': 'object',
        'properties': properties
    }

    if required:
        schema['required'] = required

    return schema


def build_output_schema(return_schema: Dict) -> Dict[str, Any]:
    """Build JSON Schema from return_schema dict."""
    if not return_schema or not isinstance(return_schema, dict):
        return {}

    properties = {}
    for field, type_val in return_schema.items():
        if isinstance(type_val, str):
            normalized = normalize_type(type_val)
            properties[field] = {
                'type': normalized,
                'description': f'Output field: {field}'
            }
        elif isinstance(type_val, dict):
            # Nested object
            properties[field] = {
                'type': 'object',
                'description': f'Output field: {field}',
                'properties': {}
            }
            for nested_field, nested_type in type_val.items():
                if isinstance(nested_type, str):
                    properties[field]['properties'][nested_field] = {
                        'type': normalize_type(nested_type),
                        'description': f'Nested field: {nested_field}'
                    }

    return {
        'type': 'object',
        'properties': properties
    }


def get_output_type(return_schema: Dict) -> str:
    """Infer output type from return_schema."""
    if not return_schema or not isinstance(return_schema, dict):
        return 'unknown'

    types_seen = set()
    for field, type_val in return_schema.items():
        if isinstance(type_val, str):
            types_seen.add(normalize_type(type_val))
        elif isinstance(type_val, dict):
            types_seen.add('object')

    if len(types_seen) == 1:
        return list(types_seen)[0]
    elif 'object' in types_seen:
        return 'dict'
    else:
        return 'mixed'


class SchemaPredictor:
    """Use LLM to predict return schemas for tools that don't have them."""

    def __init__(self, llm_client: LocalOpenAILLMClient):
        self.llm = llm_client

    def _safe_generate(self, messages: list, max_retries: int = 3) -> str:
        """Call LLM with retries."""
        import random as _rng
        for attempt in range(max_retries):
            try:
                result = self.llm.generate(messages, temperature=0.7, max_tokens=800)
                if result and result.strip():
                    return result
            except Exception as e:
                delay = min(2 * (2 ** attempt), 30) + _rng.uniform(0, 1)
                print(f"    LLM attempt {attempt + 1} failed: {e}, retrying in {delay:.1f}s...")
                time.sleep(delay)
        return "{}"

    def predict_schema(
        self,
        api_name: str,
        api_description: str,
        parameters: Dict,
        example_queries: List[str],
        max_examples: int = 3
    ) -> Dict[str, Any]:
        """Predict return schema for a tool using LLM."""

        system_prompt = """You are an expert at API documentation analysis.

Given an API's name, description, parameters, and example queries, predict what the API returns.

Respond ONLY with a valid JSON object representing the return schema.
Use format: {"field_name": "type", ...} where type is one of:
- "str" for strings
- "int" for integers
- "float" for decimal numbers
- "bool" for booleans
- "dict" for objects
- "list" for arrays

If the API returns nothing meaningful, return {}.

Examples:
- "Get user info" + params {user_id} -> {"user_id": "str", "name": "str", "email": "str"}
- "Search products" + params {query} -> {"results": [{"name": "str", "price": "float"}]}
- "Delete item" + params {id} -> {"success": "bool", "message": "str"}
"""

        params_str = json.dumps(parameters.get('properties', {}), indent=2)
        examples_str = "\n".join([f"- {q}" for q in example_queries[:max_examples]])

        user_prompt = f"""API Name: {api_name}
Description: {api_description}
Parameters:
{params_str}

Example User Queries (how this API is used):
{examples_str}

What does this API return? Respond with JSON schema only."""

        try:
            response = self._safe_generate([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            # Parse JSON response
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                response = response[start:end]

            schema = json.loads(response)
            return schema if schema else {}

        except (json.JSONDecodeError, Exception) as e:
            print(f"    Warning: Failed to parse LLM response: {e}")
            return {}


def load_toollens_corpus(corpus_path: Path) -> List[Dict[str, Any]]:
    """Load and parse corpus.jsonl."""
    tools = []

    with open(corpus_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                parsed = parse_toollens_text(entry.get('text', ''))

                if parsed:
                    tools.append(parsed)
                else:
                    print(f"Warning: Failed to parse line {line_num}")

            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}")

    return tools


def load_example_usages(toollens_data_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, Dict]]:
    """Load example usages and template_responses from ToolLens_data.json.

    Returns:
        Tuple of (examples_dict, template_dict)
        - examples_dict: maps api_name to list of example queries
        - template_dict: maps api_name to template_response (return schema)
    """
    examples = defaultdict(list)
    templates = {}

    with open(toollens_data_path) as f:
        data = json.load(f)

    for item in data:
        query = item.get('query', '')
        if not query:
            continue

        for api in item.get('apis', []):
            api_name = api.get('api_name', '')
            if not api_name:
                continue

            # Collect example queries (max 5 per API)
            if len(examples[api_name]) < 5:
                examples[api_name].append(query)

            # Collect template_response if available
            template = api.get('template_response', {})
            if template and template != {} and template != "":
                if api_name not in templates:
                    templates[api_name] = template

    return dict(examples), dict(templates)


def enhance_tool(
    tool: Dict,
    examples: Dict[str, List[str]],
    template_responses: Dict[str, Dict]
) -> Dict[str, Any]:
    """Enhance a tool with full schema information."""
    api_name = tool.get('api_name', '')
    category = tool.get('category', 'Unknown')
    tool_name = tool.get('tool_name', '')
    api_desc = tool.get('api_description', '')
    required_params = tool.get('required_params', [])
    optional_params = tool.get('optional_params', [])
    return_schema = tool.get('return_schema', {})

    # Build parameters schema
    parameters = build_parameters_schema(required_params, optional_params)

    # Build output schema from return_schema if available
    output_schema = build_output_schema(return_schema)

    # Get output type
    output_type = get_output_type(return_schema)

    # Determine if we have a schema
    has_schema = bool(return_schema)

    # Build enhanced tool
    enhanced = {
        'category': category,
        'tool_name': tool_name,
        'tool_description': f"Functions provided by the {tool_name} toolkit.",
        'api_name': api_name,
        'api_description': api_desc,
        'parameters': parameters,
        'output_type': output_type,
        'output_description': f"Returns {output_type} with fields: {', '.join(return_schema.keys()) if return_schema else 'no schema available'}",
        'output_schema': output_schema,
        'has_return_schema': has_schema,
        'schema_source': 'corpus' if has_schema else 'missing',
    }

    # Add examples if available
    if api_name in examples:
        enhanced['example_queries'] = examples[api_name]

    return enhanced, template_responses.get(api_name, {}) if api_name in template_responses else None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract tools from ToolLens dataset"
    )
    parser.add_argument(
        "--corpus",
        default="toollens_data/corpus.jsonl",
        help="Path to corpus.jsonl"
    )
    parser.add_argument(
        "--toollens-data",
        default="toollens_data/ToolLens_data.json",
        help="Path to ToolLens_data.json"
    )
    parser.add_argument(
        "--output",
        default="toollens_tools.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of tools (for testing)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM to predict missing return schemas"
    )
    parser.add_argument(
        "--model",
        default="minimax/minimax-m2.7",
        help="LLM model for schema prediction"
    )

    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    toollens_path = Path(args.toollens_data)
    output_path = Path(args.output)

    if not corpus_path.exists():
        print(f"Error: corpus not found at {corpus_path}")
        print("Download with: wget https://huggingface.co/datasets/Tool-COLT/ToolLens/resolve/main/corpus.jsonl")
        sys.exit(1)

    # Load environment
    load_dotenv()

    # Load data
    print("Loading corpus...")
    tools = load_toollens_corpus(corpus_path)
    print(f"Loaded {len(tools)} tools")

    if args.limit:
        tools = tools[:args.limit]
        print(f"Limited to {len(tools)} tools")

    # Load examples and template_responses
    examples = {}
    template_responses = {}
    if toollens_path.exists():
        print("Loading example usages and template_responses...")
        examples, template_responses = load_example_usages(toollens_path)
        print(f"Loaded examples for {len(examples)} tools")
        print(f"Loaded template_responses for {len(template_responses)} tools")
    else:
        print(f"Warning: ToolLens data not found at {toollens_path}")

    # Initialize LLM predictor if requested
    predictor = None
    if args.use_llm:
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")
        if api_key and api_base:
            print("\nInitializing LLM for schema prediction...")
            llm_client = LocalOpenAILLMClient(
                url=api_base,
                api_key=api_key,
                api_model=args.model,
                hf_tokenizer_id=None
            )
            predictor = SchemaPredictor(llm_client)
        else:
            print("Warning: OPENAI_API_KEY or OPENAI_API_BASE not set, skipping LLM prediction")

    # Phase 1: Enhance tools with schemas from corpus
    print("\nPhase 1: Enhancing tools...")
    enhanced_tools = []
    tools_need_schema = []  # Track tools that need schema filling

    for i, tool in enumerate(tools, 1):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(tools)}")

        enhanced, template_response = enhance_tool(tool, examples, template_responses)
        api_name = enhanced['api_name']

        # If tool doesn't have return_schema but we have template_response from ToolLens_data
        if not enhanced['has_return_schema'] and template_response:
            enhanced['return_schema'] = template_response
            enhanced['output_schema'] = build_output_schema(template_response)
            enhanced['output_type'] = get_output_type(template_response)
            enhanced['has_return_schema'] = True
            enhanced['schema_source'] = 'template_response'
        elif not enhanced['has_return_schema']:
            tools_need_schema.append((len(enhanced_tools), enhanced))
            enhanced['return_schema'] = {}

        enhanced_tools.append(enhanced)

    # Phase 2: Fill remaining missing schemas with LLM
    if predictor and tools_need_schema:
        print(f"\nPhase 2: LLM predicting schemas for {len(tools_need_schema)} tools...")

        for idx, enhanced in tools_need_schema:
            api_name = enhanced['api_name']
            api_desc = enhanced['api_description']
            params = enhanced['parameters']
            ex_queries = enhanced.get('example_queries', [])

            print(f"  Predicting schema for: {api_name}")

            # Try up to 3 times
            for attempt in range(3):
                predicted_schema = predictor.predict_schema(
                    api_name=api_name,
                    api_description=api_desc,
                    parameters=params,
                    example_queries=ex_queries
                )

                if predicted_schema:
                    break
                print(f"    Attempt {attempt + 1} failed, retrying...")

            if predicted_schema:
                enhanced['return_schema'] = predicted_schema
                enhanced['output_schema'] = build_output_schema(predicted_schema)
                enhanced['output_type'] = get_output_type(predicted_schema)
                enhanced['has_return_schema'] = True
                enhanced['schema_source'] = 'llm_predicted'
            else:
                enhanced['schema_source'] = 'llm_failed'

            enhanced_tools[idx] = enhanced

    # Save
    print(f"\nSaving {len(enhanced_tools)} tools to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for tool in enhanced_tools:
            f.write(json.dumps(tool, ensure_ascii=False) + '\n')

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Count by category
    categories = defaultdict(int)
    for tool in enhanced_tools:
        categories[tool['category']] += 1

    print("\nTools by category:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:30s}: {count:3d}")

    # Count by schema source
    schema_sources = defaultdict(int)
    for tool in enhanced_tools:
        source = tool.get('schema_source', 'unknown')
        schema_sources[source] += 1

    print("\nSchema sources:")
    for source, count in sorted(schema_sources.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source:20s}: {count:3d}")

    # Count with examples
    with_examples = sum(1 for t in enhanced_tools if t.get('example_queries'))
    print(f"\nWith example queries: {with_examples}/{len(enhanced_tools)}")

    print("\n" + "=" * 60)
    print(f"Output saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()