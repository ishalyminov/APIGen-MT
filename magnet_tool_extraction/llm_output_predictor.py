"""
LLM-based schema predictor for tool extraction.

Provides unified schema prediction for both BFCL and ToolLens datasets.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

from llm_client import LocalOpenAILLMClient


class OutputPrediction(BaseModel):
    """Schema for LLM output prediction"""
    output_type: str = "unknown"
    output_description: str = ""
    return_schema: Optional[Dict[str, Any]] = None


class SchemaPredictor:
    """
    Unified LLM-based schema predictor for both BFCL and ToolLens tools.

    Supports two modes:
    - "bfcl": Returns output_type and output_description (for BFCL-style tools)
    - "toollens": Returns return_schema dict mapping field names to types (for ToolLens-style tools)

    Uses LocalOpenAILLMClient which is the standard LLM client for this codebase.
    """

    def __init__(self, model: str = "minimax/minimax-m2.7", debug: bool = False,
                 api_key: str = None, api_base: str = None):
        """
        Initialize the schema predictor.

        Args:
            model: Model name for LocalOpenAILLMClient
            debug: Enable debug logging
            api_key: Optional API key (falls back to OPENAI_API_KEY env var)
            api_base: Optional API base URL (falls back to OPENAI_API_BASE env var)
        """
        self.debug = debug

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        api_base = api_base or os.getenv("OPENAI_API_BASE")

        if not api_key or not api_base:
            raise ValueError("OPENAI_API_KEY and OPENAI_API_BASE must be set")

        self.llm = LocalOpenAILLMClient(
            url=api_base,
            api_key=api_key,
            api_model=model,
            hf_tokenizer_id=None
        )
        self.model = model

    def _safe_generate(self, messages: list, max_retries: int = 3, max_tokens: int = 800) -> str:
        """Call LLM with retries."""
        import random as _rng
        for attempt in range(max_retries):
            try:
                result = self.llm.generate(messages, temperature=0.7, max_tokens=max_tokens)
                if result and result.strip():
                    return result
                raise ValueError("Empty response")
            except Exception as e:
                delay = min(2 * (2 ** attempt), 30) + _rng.uniform(0, 1)
                print(f"    LLM attempt {attempt + 1} failed: {e}, retrying in {delay:.1f}s...")
                time.sleep(delay)
        return "{}"

    def predict_for_bfcl(
        self,
        tool_schema: Dict[str, Any],
        invocation_contexts: List[Dict[str, Any]],
        max_contexts: int = 5,
    ) -> OutputPrediction:
        """
        Predict output_type and output_description (BFCL style).

        Args:
            tool_schema: Tool definition schema
            invocation_contexts: List of invocation contexts with user_message, assistant_message, tool_calls
            max_contexts: Maximum number of contexts to include

        Returns:
            OutputPrediction with output_type and output_description
        """
        system_prompt = """You are an expert at analyzing function/tool schemas and predicting what they return.

Given information about a function/tool:
1. Its name and description
2. Its parameters (arguments it accepts)
3. Example invocations showing how it's used in practice

Your task is to predict:
1. **output_type**: The type of data the function returns (e.g., "string", "integer", "boolean", "dict", "list", "file content", "API response", "operation status", etc.)
2. **output_description**: A clear description of what the function returns, including the structure if it's a complex type

Guidelines:
- Be specific about the output type (e.g., "weather data dict" instead of just "dict")
- Include important fields if returning a structured type
- Mention if the function returns success/failure status
- Consider what makes sense given the function's purpose and parameters
- Use the invocation contexts to understand real-world usage patterns

Respond ONLY with a valid JSON object matching the schema:
{
  "output_type": "string",
  "output_description": "string"
}}"""

        # Build prompt
        tool_name = tool_schema.get('tool_name', 'unknown')
        api_name = tool_schema.get('api_name', 'unknown')
        api_description = tool_schema.get('api_description', 'No description available')
        parameters = tool_schema.get('parameters', {})

        prompt_parts = [
            f"# Function Information\n",
            f"**Tool Name**: {tool_name}",
            f"**API Name**: {api_name}",
            f"**Description**: {api_description}",
            f"",
        ]

        if parameters:
            prompt_parts.append("## Parameters\n")
            props = parameters.get('properties', {})
            if hasattr(parameters, 'to_dict'):
                parameters = parameters.to_dict()
            prompt_parts.append(f"```json\n{json.dumps(parameters, indent=2)}\n```\n")

        # Include output schema if available (for ToolLens tools)
        output_schema = tool_schema.get('output_schema', {})
        if output_schema and output_schema.get('properties'):
            prompt_parts.append("## Expected Output Schema\n")
            prompt_parts.append(f"```json\n{json.dumps(output_schema, indent=2)}\n```\n")

        if invocation_contexts:
            prompt_parts.append("## Example Invocations\n")
            for i, ctx in enumerate(invocation_contexts[:max_contexts], 1):
                user_message = ctx.get('user_message', 'N/A')
                assistant_message = ctx.get('assistant_message', 'N/A')
                tool_calls = ctx.get('tool_calls', [])

                prompt_parts.append(f"### Context {i}\n")
                prompt_parts.append(f"**User**: {user_message}\n")
                prompt_parts.append(f"**Assistant**: {assistant_message}\n")

                if tool_calls:
                    prompt_parts.append("**Tool Calls**:\n")
                    for tc in tool_calls:
                        prompt_parts.append(f"- {tc.get('name', 'unknown')}: {json.dumps(tc.get('arguments', {}))}\n")

                prompt_parts.append("")

        prompt_parts.extend([
            "## Task\n",
            "Based on the function information and example invocations above, predict:\n",
            "1. The output type this function returns\n",
            "2. A clear description of what the output contains\n",
            "\nRespond with a JSON object containing 'output_type' and 'output_description'.",
        ])

        prompt = "\n".join(prompt_parts)

        if self.debug:
            print(f"\nPredicting for {api_name}")

        try:
            response = self._safe_generate([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])

            # Parse JSON
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                response = response[start:end]

            pred_dict = json.loads(response)
            return OutputPrediction(
                output_type=pred_dict.get('output_type', 'unknown'),
                output_description=pred_dict.get('output_description', ''),
                return_schema=None
            )

        except Exception as e:
            print(f"    Warning: Failed to parse prediction: {e}")
            return OutputPrediction(output_type="unknown", output_description="Failed to predict")

    def predict_for_toollens(
        self,
        api_name: str,
        api_description: str,
        parameters: Dict[str, Any],
        example_queries: List[str],
        max_examples: int = 3,
    ) -> Dict[str, Any]:
        """
        Predict return_schema (ToolLens style).

        Returns a dict mapping field names to types, e.g., {"success": "bool", "message": "str"}.

        Args:
            api_name: The API function name
            api_description: What the API does
            parameters: Parameter schema
            example_queries: Example user queries showing how the API is used
            max_examples: Max examples to include

        Returns:
            Dict mapping field names to type strings
        """
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
- "Get random fact" + params {type} -> {"fact": "str", "type": "str"}
- "Airport data" + params {} -> {"airports": [{"name": "str", "code": "str", "city": "str"}]}"""

        params_str = json.dumps(parameters.get('properties', {}), indent=2)
        examples_str = "\n".join([f"- {q}" for q in example_queries[:max_examples]])

        user_prompt = f"""API Name: {api_name}
Description: {api_description}
Parameters:
{params_str}

Example User Queries (how this API is used):
{examples_str}

What does this API return? Respond with JSON schema only."""

        if self.debug:
            print(f"\nPredicting schema for {api_name}")

        try:
            response = self._safe_generate([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], max_tokens=600)

            # Parse JSON
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                response = response[start:end]

            return json.loads(response)

        except Exception as e:
            print(f"    Warning: Failed to predict schema: {e}")
            return {}

    # Backwards compatibility alias
    def predict_output(self, tool_schema: Dict[str, Any], invocation_contexts: List[Dict[str, Any]], max_contexts: int = 5) -> OutputPrediction:
        """Alias for predict_for_bfcl (backwards compatibility)."""
        return self.predict_for_bfcl(tool_schema, invocation_contexts, max_contexts)


# ─── Convenience functions ────────────────────────────────────────────────────

def predict_outputs_for_tools(
    tools: List[Dict[str, Any]],
    invocations: List[Dict[str, Any]],
    model: str = "minimax/minimax-m2.7",
    max_contexts: int = 5,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Predict output types and descriptions for multiple tools (BFCL style).

    Args:
        tools: List of tool definitions
        invocations: List of all invocation examples
        model: LLM model to use
        max_contexts: Maximum invocation contexts per tool
        debug: Enable debug mode

    Returns:
        List of tool definitions with added output_type and output_description
    """
    predictor = SchemaPredictor(model=model, debug=debug)

    # Index invocations by tool name
    invocations_by_tool = {}
    for inv in invocations:
        tool_name = inv.get('tool_name')
        if tool_name not in invocations_by_tool:
            invocations_by_tool[tool_name] = []
        invocations_by_tool[tool_name].append(inv)

    enhanced_tools = []

    print(f"\n{'='*80}")
    print("🤖 PREDICTING OUTPUTS USING LLM")
    print(f"{'='*80}")
    print(f"Max contexts per tool: {max_contexts}")
    print(f"Debug mode: {debug}")
    print()

    for i, tool in enumerate(tools, 1):
        tool_name = tool.get('tool_name')
        api_name = tool.get('api_name', 'unknown')

        print(f"Processing tool {i}/{len(tools)}: {api_name}")

        # Get invocation contexts for this tool
        tool_invocations = invocations_by_tool.get(tool_name, [])

        # Predict output
        prediction = predictor.predict_for_bfcl(tool, tool_invocations, max_contexts)

        # Create enhanced tool definition
        enhanced_tool = {
            **tool,
            'output_type': prediction.output_type,
            'output_description': prediction.output_description
        }

        enhanced_tools.append(enhanced_tool)

    return enhanced_tools


def predict_toollens_schemas(
    tools: List[Dict[str, Any]],
    examples: Dict[str, List[str]],
    model: str = "minimax/minimax-m2.7",
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Predict return schemas for ToolLens tools.

    Args:
        tools: List of tool definitions
        examples: Dict mapping api_name to list of example queries
        model: LLM model to use
        debug: Enable debug mode

    Returns:
        List of tool definitions with added return_schema
    """
    predictor = SchemaPredictor(model=model, debug=debug)

    enhanced_tools = []

    print(f"\n{'='*80}")
    print("🤖 PREDICTING TOOLLENS SCHEMAS USING LLM")
    print(f"{'='*80}")
    print()

    for i, tool in enumerate(tools, 1):
        api_name = tool.get('api_name', 'unknown')

        print(f"Processing tool {i}/{len(tools)}: {api_name}")

        # Get example queries for this tool
        tool_examples = examples.get(api_name, [])

        # Predict schema
        return_schema = predictor.predict_for_toollens(
            api_name=api_name,
            api_description=tool.get('api_description', ''),
            parameters=tool.get('parameters', {}),
            example_queries=tool_examples
        )

        # Create enhanced tool definition
        enhanced_tool = {
            **tool,
            'return_schema': return_schema
        }

        enhanced_tools.append(enhanced_tool)

    return enhanced_tools


# ─── Backwards compatibility aliases ─────────────────────────────────────────

# Keep old class name as alias
LLMOutputPredictor = SchemaPredictor


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Test the predictor
    test_tool = {
        "tool_name": "weather_api",
        "api_name": "get_weather",
        "api_description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or zip code"
                }
            },
            "required": ["location"]
        }
    }

    test_contexts = [
        {
            "user_message": "What's the weather in San Francisco?",
            "assistant_message": "Let me check the weather for you.",
            "tool_calls": [
                {
                    "name": "get_weather",
                    "arguments": {"location": "San Francisco"}
                }
            ]
        }
    ]

    predictor = SchemaPredictor(debug=True)
    prediction = predictor.predict_for_bfcl(test_tool, test_contexts)

    print(f"\n{'='*80}")
    print("BFCL PREDICTION RESULT")
    print(f"{'='*80}")
    print(f"Output Type: {prediction.output_type}")
    print(f"Output Description: {prediction.output_description}")

    print(f"\n{'='*80}")
    print("TOOLLENS SCHEMA PREDICTION")
    print(f"{'='*80}")
    schema = predictor.predict_for_toollens(
        api_name="Airport data",
        api_description="API returns a file with a list of airports from the database",
        parameters={"type": "object", "properties": {}},
        example_queries=["Show me airports in California", "List all airports"]
    )
    print(f"Return Schema: {schema}")