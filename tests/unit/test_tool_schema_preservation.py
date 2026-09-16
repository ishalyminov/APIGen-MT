import json

from tool_manager import ToolManager


def test_bfcl_parameter_constraints_are_preserved(tmp_path):
    tool_file = tmp_path / "tools.jsonl"
    tool_file.write_text(
        json.dumps(
            {
                "api_name": "demo_tool",
                "category": "Demo",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["fast", "safe"],
                            "pattern": "^[a-z]+$",
                        },
                        "values": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "float",
                                "minimum": 0,
                            },
                        },
                    },
                    "required": ["mode", "values"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manager = ToolManager(llm=None, tool_pool_path=str(tool_file), use_config_pool=False)
    params = manager.get_tool_schema("demo_tool")["parameters"]

    assert params["properties"]["mode"]["enum"] == ["fast", "safe"]
    assert params["properties"]["mode"]["pattern"] == "^[a-z]+$"
    assert params["properties"]["values"]["minItems"] == 1
    assert params["properties"]["values"]["items"] == {
        "type": "number",
        "minimum": 0,
    }
