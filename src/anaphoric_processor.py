#!/usr/bin/env python3
"""
Post-process generated datapoints to make them more anaphoric.

If something is mentioned in previous dialog turns, the system should refer to it
using references (pronouns, partial mentions, "it", "the X from before") instead of
duplicating the exact word or phrase.

Example transformations:
- "Book a flight from New York to Los Angeles" → "Book that flight to LAX"
- "Search for messages containing 'meeting'" → "Search for those messages"
- "The booking ID {{TURN1.book_flight.booking_id}}" → "The {{TURN1.book_flight.booking_id}}"
"""

import json
import os
import sys

# Fix path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))
sys.path.insert(0, script_dir)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from llm_client import LocalOpenAILLMClient
from dotenv import load_dotenv


def load_datapoints(input_path: str) -> List[Dict]:
    """Load datapoints from JSONL file."""
    datapoints = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                datapoints.append(json.loads(line))
    return datapoints


def save_datapoints(datapoints: List[Dict], output_path: str):
    """Save datapoints to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for dp in datapoints:
            f.write(json.dumps(dp, ensure_ascii=False) + '\n')


def build_anaphoric_prompt(conversation_history: str, current_query: str, turn_number: int) -> str:
    """Build prompt for making a query more anaphoric by removing explicit values."""
    return f"""You are editing multi-turn queries to remove explicit values and make the model rely on dialog history.

PREVIOUS TURNS (the model will see tool calls and outputs):
{conversation_history}

CURRENT QUERY (rewrite to remove explicit values):
{current_query}

CRITICAL INSTRUCTIONS:
1. REMOVE ALL explicit values that appear in the previous conversation
2. Replace computed results with "the calculated [result name]"
3. Replace IDs with "that [entity]"
4. Do NOT repeat values like 800.0, 150, message_id 5, etc.
5. If a value is in the history above, remove it and refer indirectly

EXAMPLES OF TRANSFORMATIONS:
- "use the sum (800.0) from turn 1" → "use the calculated sum"
- "delete message with ID 5" → "delete that message"
- "absolute value of 800.0" → "absolute value of the result"
- "find square root of that sum" → "find square root of that"
- "book flight to LAX on 2024-12-20" → "book that flight"

IMPORTANT: Do NOT keep explicit values in parentheses. Remove them entirely.

Respond with ONLY the rewritten query, nothing else."""


def process_turn_query(
    client: LocalOpenAILLMClient,
    conversation_history: str,
    current_query: str,
    turn_number: int,
    max_retries: int = 2
) -> str:
    """Process a single turn query to make it more anaphoric."""
    prompt = build_anaphoric_prompt(conversation_history, current_query, turn_number)

    for attempt in range(max_retries):
        try:
            response = client.generate([{"role": "user", "content": prompt}])
            if response and response.strip():
                return response.strip()
        except Exception as e:
            print(f"    [Attempt {attempt + 1}/{max_retries}] Error: {e}")
            import time
            time.sleep(2 ** attempt)

    # Fallback: return original if all retries fail
    print(f"    [WARNING] Using original query after {max_retries} retries")
    return current_query


def format_turn_history(turns: List[Dict], up_to_turn: int) -> str:
    """Format conversation history for prompts."""
    lines = []
    for i, turn in enumerate(turns[:up_to_turn]):
        turn_num = turn.get('turn_number', i + 1)
        query = turn.get('user_query', '')
        lines.append(f"Turn {turn_num}: {query}")

        for step in turn.get('steps', []):
            for tc in step.get('tool_calls', []):
                tool_name = tc.get('tool_name', '')
                args = tc.get('arguments', {})
                output = tc.get('output', {})
                output_str = json.dumps(output, default=str)[:150] if output else ''
                lines.append(f"  Tool: {tool_name}")
                lines.append(f"    Args: {json.dumps(args)}")
                if output:
                    lines.append(f"    Output: {output_str}")

    return '\n'.join(lines)


def process_datapoint(client: LocalOpenAILLMClient, dp: Dict, dry_run: bool = False) -> Dict:
    """Process a single datapoint to make queries more anaphoric."""
    if 'conversation' not in dp or 'turns' not in dp['conversation']:
        return dp

    conversation = dp['conversation']
    turns = conversation['turns']

    print(f"  Processing {len(turns)} turns...")

    # Process each turn's query (skip turn 1, it has no prior context)
    for turn_idx in range(1, len(turns)):
        turn = turns[turn_idx]
        original_query = turn.get('user_query', '')

        if not original_query:
            continue

        # Build conversation history up to this turn
        history = format_turn_history(turns, turn_idx)

        # Get anaphoric version
        if not dry_run:
            try:
                new_query = process_turn_query(
                    client,
                    history,
                    original_query,
                    turn.get('turn_number', turn_idx + 1)
                )

                if new_query != original_query:
                    print(f"    Turn {turn_idx + 1}: '{original_query[:40]}...' → '{new_query[:40]}...'")
                    turn['user_query'] = new_query
                    turn['user_query_anaphoric'] = original_query
            except Exception as e:
                print(f"    Turn {turn_idx + 1}: Error processing - {e}")
        else:
            print(f"    Turn {turn_idx + 1} (dry run): would transform '{original_query[:40]}...'")

    return dp


def process_datapoints_batch(
    client: LocalOpenAILLMClient,
    datapoints: List[Dict],
    batch_size: int = 5
) -> List[Dict]:
    """Process datapoints in batches with progress reporting."""
    processed = []

    for i, dp in enumerate(datapoints):
        task = dp.get('conversation', {}).get('overall_task', '?')[:50]
        print(f"\n[{i+1}/{len(datapoints)}] Processing: {task}...")
        try:
            processed_dp = process_datapoint(client, dp)
            processed.append(processed_dp)
            print(f"  ✓ Done")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            processed.append(dp)  # Keep original on error

    return processed


def main():
    parser = argparse.ArgumentParser(description='Post-process datapoints to make them more anaphoric')
    parser.add_argument('input', help='Input JSONL file with datapoints')
    parser.add_argument('output', help='Output JSONL file for processed datapoints')
    parser.add_argument('--batch-size', '-b', type=int, default=5, help='Batch size for processing')
    parser.add_argument('--model', '-m', type=str, default='minimaxai/minimax-m2.7', help='Model to use')
    parser.add_argument('--dry-run', action='store_true', help='Show what would change without making changes')
    parser.add_argument('--num-datapoints', '-n', type=int, default=None, help='Limit number of datapoints to process')
    return parser.parse_args()


if __name__ == '__main__':
    args = main()
    print("Arguments parsed", flush=True)

    # Load environment
    load_dotenv()
    print("Environment loaded", flush=True)

    # Initialize LLM client
    api_key = os.getenv('OPENAI_API_KEY')
    api_base = os.getenv('OPENAI_API_BASE')
    print(f"API: {api_base}, key present: {bool(api_key)}", flush=True)

    if not api_key or not api_base:
        print("ERROR: OPENAI_API_KEY or OPENAI_API_BASE not set")
        sys.exit(1)

    client = LocalOpenAILLMClient(
        url=api_base,
        api_key=api_key,
        api_model=args.model,
        hf_tokenizer_id=None,
    )

    # Load datapoints
    print(f"Loading datapoints from {args.input}...")
    datapoints = load_datapoints(args.input)

    if args.num_datapoints:
        datapoints = datapoints[:args.num_datapoints]

    print(f"Loaded {len(datapoints)} datapoints")

    if args.dry_run:
        print("\n=== DRY RUN MODE ===")

    # Process datapoints
    processed = process_datapoints_batch(client, datapoints, args.batch_size)

    # Save results
    if not args.dry_run:
        print(f"\nSaving processed datapoints to {args.output}...")
        save_datapoints(processed, args.output)
        print("Done!")
    else:
        print("\nDry run complete - no changes saved")