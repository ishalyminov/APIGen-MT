# Anaphoric Processing Results

This document compares original user queries with their anaphoric-processed versions. The goal is to make queries rely on dialog history instead of repeating explicit values.

## Overview

The anaphoric processor transforms multi-turn conversation queries to:
1. **Remove explicit values** that were mentioned/computed in previous turns
2. **Replace with references** to the dialog history
3. **Make the model rely on context** rather than having values spoon-fed

## Transformation Examples

### Example 1: Stock Price Calculation

| Turn | Before | After |
|------|--------|-------|
| 1 | I have the closing prices for AAPL stock on 2023-01-01: $150, $155, $160, $165, $170. Please compute the sum and the mean of these prices. | I have the closing prices for AAPL stock on 2023-01-01: $150, $155, $160, $165, $170. Please compute the sum and the mean of these prices. |
| 2 | For record ID 98765, use the sum from the previous calculation (800.0) to compute its absolute value and then find the square root of that sum. | For that record, use the calculated sum to compute its absolute value and then find the square root of that. |

**Changes in Turn 2:**
- "For record ID 98765" → "For that record" (anaphoric reference)
- "use the sum from the previous calculation (800.0)" → "use the calculated sum" (removed explicit value)
- "find the square root of that sum" → "find the square root of that" (removed "sum")

---

### Example 2: Twitter Engagement

| Turn | Before | After |
|------|--------|-------|
| 1 | tech_user/TechUser2024! authentication. Post tweet 'Our Q4 2024 roadmap is live!... | tech_user/TechUser2024! authentication. Post tweet 'Our Q4 2024 roadmap is live!... |
| 2 | Get the tweet we just posted and add a comment 'Thanks for the overwhelming response!... | Get that tweet and add a comment 'Thanks for the overwhelming response!... |

**Changes in Turn 2:**
- "Get the tweet we just posted" → "Get that tweet" (removed "we just posted", use pronoun)

---

## Transformation Types

1. **Value removal**: "(800.0)" was removed entirely
2. **Pronoun substitution**: "the sum from the previous calculation" → "the calculated sum"
3. **Entity shortening**: "For record ID 98765" → "For that record"
4. **Verb naturalization**: "the tweet we just posted" → "that tweet"

## Files

- `src/anaphoric_processor.py` - The LLM-based post-processing script
- `data/anaphoric/quick_output_v3.jsonl` - Processed datapoints demonstrating the concept

## Usage

```bash
# Process datapoints
python src/anaphoric_processor.py input.jsonl output.jsonl

# Preview changes without applying
python src/anaphoric_processor.py input.jsonl output.jsonl --dry-run
```

## Design Goals

The post-processing ensures that:
1. Explicit values from previous turns are not repeated
2. Queries reference the dialog history naturally
3. The model must track state/context to understand what values were computed
4. Credentials and auth info are preserved when needed for tool execution