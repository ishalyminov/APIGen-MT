#!/usr/bin/env python3
"""Generate and verify Communication datapoint."""
import sys
import os
import json
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, 'src')

from llm_client import LocalOpenAILLMClient
from tool_manager import ToolManager
from apigen_multi_turn import MultiTurnGenerator

api_key = os.getenv('OPENAI_API_KEY')
api_base = os.getenv('OPENAI_API_BASE')
model = os.getenv('LLM_MODEL', 'minimax/minimax-m2.7')

llm_client = LocalOpenAILLMClient(
    url=api_base,
    api_key=api_key,
    api_model=model,
    hf_tokenizer_id=None,
)

tool_manager = ToolManager(
    llm=llm_client,
    tool_pool_path='magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl',
    invocation_examples_path='magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl',
)

generator = MultiTurnGenerator(
    llm_client=llm_client,
    tool_manager=tool_manager,
    num_turns=3,
    actions_per_turn=2,
)

print('Generating Communication datapoint...')
dp = generator.generate_multi_turn_datapoint(focus_category='Communication')

if dp:
    print('\n=== GENERATED DATAPOINT ===')
    for i, turn in enumerate(dp.conversation.turns, 1):
        print(f'\n--- Turn {i} ---')
        print(f'Query: {turn.user_query}')
        print(f'Tools: {turn.expected_tools}')
        for step in turn.steps:
            for tc in step.tool_calls:
                print(f'  {tc.tool_name}({json.dumps(tc.arguments)}) -> {json.dumps(tc.output)[:150]}')
else:
    print('Generation failed!')
    sys.exit(1)

# Exhaustive verification
print('\n' + '='*70)
print('EXPENSIVE VERIFICATION')
print('='*70)

# Initialize fresh tool_manager for verification
tool_manager2 = ToolManager(
    llm=llm_client,
    tool_pool_path='magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl',
    invocation_examples_path='magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl',
)
tool_manager2.initialize_api_state()

msg_api = tool_manager2.python_tool_instances.get('message_api')
if not msg_api:
    print('ERROR: message_api not found')
    sys.exit(1)

print(f'\nInitial user_map: {msg_api.user_map}')
print(f'Initial current_user: {msg_api.current_user}')
print(f'Initial user_count: {msg_api.user_count}')

# Get initial state
api_state = tool_manager2.get_api_state()
msg_state = api_state.get('message_api', {})
print(f'\nInitial message_count: {msg_state.get("message_count", "N/A")}')
print(f'Initial messages_sent_map: {json.dumps(msg_state.get("messages_sent_map", {}), default=str)[:200]}')

# Simulate each turn
print('\n=== SIMULATING TURNS ===')

for turn_idx, turn in enumerate(dp.conversation.turns, 1):
    print(f'\n--- Turn {turn_idx} ---')
    print(f'Query: {turn.user_query}')
    print(f'Tools: {turn.expected_tools}')

    for step in turn.steps:
        for tc in step.tool_calls:
            print(f'\n  Executing {tc.tool_name} with args {tc.arguments}...')
            try:
                result = tool_manager2.invoke_python_tool(tc.tool_name, tc.arguments)
                print(f'  Result: {json.dumps(result, default=str)[:200]}')

                # Check for errors
                if isinstance(result, dict):
                    if result.get('login_status') is False:
                        print(f'  WARNING: Login failed')
                    if result.get('sent_status') is False:
                        print(f'  WARNING: Send failed')
                    if result.get('deleted_status') is False:
                        print(f'  WARNING: Delete failed')
                    if result.get('added_status') is False:
                        print(f'  WARNING: Add failed')
                    if 'error' in result:
                        print(f'  ERROR: {result["error"]}')

            except Exception as e:
                print(f'  EXCEPTION: {e}')

# Final state check
final_state = tool_manager2.get_api_state()
msg_final = final_state.get('message_api', {})
print(f'\n=== FINAL STATE ===')
print(f'Message count: {msg_final.get("message_count", "N/A")}')
print(f'Current user: {msg_final.get("current_user", "N/A")}')

# Check what messages were actually sent
sent_map = msg_final.get('messages_sent_map', {})
print(f'Messages sent: {len(sent_map)} users have sent messages')
for user, messages in sent_map.items():
    print(f'  {user}: {len(messages)} recipients')

inbox_map = msg_final.get('messages_inbox_map', {})
print(f'Messages received: {len(inbox_map)} users have messages')
for user, messages in inbox_map.items():
    total = sum(len(msgs) for msgs in messages.values())
    print(f'  {user}: {total} messages')

print('\n=== VERDICT ===')
errors_found = False

# Check for error indicators in outputs
for turn_idx, turn in enumerate(dp.conversation.turns, 1):
    for step in turn.steps:
        for tc in step.tool_calls:
            output = tc.output
            if isinstance(output, dict):
                if output.get('login_status') is False:
                    print(f'ERROR: Turn {turn_idx} {tc.tool_name} - login failed: {output.get("message")}')
                    errors_found = True
                if output.get('sent_status') is False:
                    print(f'ERROR: Turn {turn_idx} {tc.tool_name} - send failed: {output.get("message")}')
                    errors_found = True
                if output.get('deleted_status') is False:
                    print(f'ERROR: Turn {turn_idx} {tc.tool_name} - delete failed: {output.get("message")}')
                    errors_found = True
                if output.get('added_status') is False:
                    print(f'ERROR: Turn {turn_idx} {tc.tool_name} - add failed: {output.get("message")}')
                    errors_found = True
                if 'error' in output:
                    print(f'ERROR: Turn {turn_idx} {tc.tool_name} - error: {output.get("error")}')
                    errors_found = True

if not errors_found:
    print('All tool calls succeeded without errors!')
else:
    print('Errors found - see above.')