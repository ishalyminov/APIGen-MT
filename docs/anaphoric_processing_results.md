# Anaphoric Processing Results

This document compares original user queries with their anaphoric-processed versions.

## Overview

The anaphoric processor transforms multi-turn conversation queries to use natural language references instead of repeating exact phrases. This makes the conversations more realistic and human-like.

## Transformation Examples

### Example 1: Stock Price Calculation

| Turn | Before | After |
|------|--------|-------|
| 1 | I have the closing prices for AAPL stock on 2023-01-01: $150, $155, $160, $165, $170. Please compute the sum and the mean of these prices. | I have the closing prices for AAPL stock on 2023-01-01: $150, $155, $160, $165, $170. Please compute the sum and the mean of these prices. |
| 2 | For record ID 98765, use the sum from the previous calculation (800.0) to compute its absolute value and then find the square root of that sum. | For record 98765, compute the absolute value of that sum (800.0) and then find its square root. |

**Changes in Turn 2:**
- "For record ID 98765" → "For record 98765" (shortened)
- "use the sum from the previous calculation (800.0)" → "compute the absolute value of that sum (800.0)" (anaphoric reference)
- "find the square root of that sum" → "find its square root" (pronoun instead of repeating "sum")

---

### Example 2: Twitter Engagement

| Turn | Before | After |
|------|--------|-------|
| 1 | tech_user/TechUser2024! authentication. Post tweet 'Our Q4 2024 roadmap is live! 🚀 New features include real-time analytics (launching 2024-03-01) and API v3.0 (launching 2024-04-15). Full details: https://example.com/roadmap' | tech_user/TechUser2024! authentication. Post tweet 'Our Q4 2024 roadmap is live! 🚀 New features include real-time analytics (launching 2024-03-01) and API v3.0 (launching 2024-04-15). Full details: https://example.com/roadmap' |
| 2 | Get the tweet we just posted and add a comment 'Thanks for the overwhelming response! Team is working around the clock to deliver these features on schedule.' | Retrieve that tweet and add a comment 'Thanks for the overwhelming response! Team is working around the clock to deliver these features on schedule.' |

**Changes in Turn 2:**
- "Get the tweet we just posted" → "Retrieve that tweet" (anaphoric reference instead of "the tweet we just posted")
- Other content preserved to maintain tool execution requirements

---

## Summary of Transformation Types

1. **Pronoun substitution**: "the sum from the previous calculation" → "that sum"
2. **Pronoun shortening**: "find the square root of that sum" → "find its square root"
3. **Entity shortening**: "For record ID 98765" → "For record 98765"
4. **Verb naturalization**: "Get the tweet we just posted" → "Retrieve that tweet"

## Files

- `src/anaphoric_processor.py` - The LLM-based post-processing script
- `data/anaphoric/` - Processed datapoints

## Usage

```bash
python src/anaphoric_processor.py input.jsonl output.jsonl
```

Add `--dry-run` to preview changes without applying them.

## Notes

- Only turns after the first turn are processed (first turn has no prior context)
- Original queries are preserved in `user_query_anaphoric` field for reference
- Critical values (IDs, credentials) are preserved to ensure tool execution works
- Rate limiting from the NVIDIA API may cause processing delays