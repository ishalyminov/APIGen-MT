# Anaphoric Processing Results

This document compares original user queries with their anaphoric-processed versions. The goal is to make queries rely on dialog history instead of repeating explicit values.

## Overview

The anaphoric processor transforms multi-turn conversation queries to:
1. **Remove explicit values** that were mentioned/computed in previous turns
2. **Replace with references** to the dialog history
3. **Make the model rely on context** rather than having values spoon-fed

## Transformation Examples

### Example 1: Twitter - Post and Retrieve Tweet

| Turn | Before | After |
|------|--------|-------|
| 1 | Authenticate me on Twitter as tech_user with password TechUser2024! and then post a tweet saying 'Just read an amazing article about AI...' | Authenticate me on Twitter as tech_user with password TechUser2024! and then post a tweet saying 'Just read an amazing article about AI...' |
| 2 | Now retrieve the tweet I just posted about the new AI feature and retweet it for my followers. | Now retrieve that tweet about the new AI feature and retweet it for my followers. |

**Changes:** "the tweet I just posted" → "that tweet"

---

### Example 2: Vehicle Control - Road Trip Planning

| Turn | Before | After |
|------|--------|-------|
| 1 | Check the full status of my 2024 Tesla Model 3 and then set navigation to the office at 450 Mission Street. | Check the full status of my 2024 Tesla Model 3 and then set navigation to the office at 450 Mission Street. |
| 2 | Based on the current fuel level and the distance to San Francisco, can I make the trip? Also show me the distance from my current location to that destination. | Based on the fuel level and the distance to that destination, can I make the trip? Also show me the distance from my current location to that destination. |

**Changes:**
- "Based on the current fuel level" → "Based on the fuel level"
- "the distance to San Francisco" → "the distance to that destination"

---

### Example 3: Ticketing System - Resolve Ticket

| Turn | Before | After |
|------|--------|-------|
| 1 | Log into the ticket system as support_agent with password SupportAgent2024! and create a ticket titled 'Server outage' with critical priority. | Log into the ticket system as support_agent with password SupportAgent2024! and create a ticket titled 'Server outage' with critical priority. |
| 2 | Resolve ticket 654322 with resolution 'Restarted the server successfully' and then close it. | Resolve that ticket with the resolution details and then close it. |

**Changes:**
- "ticket 654322" → "that ticket"
- "'Restarted the server successfully'" → "the resolution details"

---

### Example 4: Twitter - Retweet and Follow

| Turn | Before | After |
|------|--------|-------|
| 1 | Log me into Twitter as tech_user with password TechUser2024! and post a tweet saying 'Just read an amazing article...' | Log me into Twitter as tech_user with password TechUser2024! and post a tweet saying 'Just read an amazing article...' |
| 2 | Now retweet that last tweet I posted and follow @TechCrunch to stay updated on tech industry news. | Now retweet the last tweet and follow @TechCrunch to stay updated on tech industry news. |

**Changes:** "that last tweet I posted" → "the last tweet"

---

### Example 5: Stock Price Calculation

| Turn | Before | After |
|------|--------|-------|
| 1 | I have the closing prices for AAPL stock on 2023-01-01: $150, $155, $160, $165, $170. Please compute the sum and the mean of these prices. | I have the closing prices for AAPL stock on 2023-01-01: $150, $155, $160, $165, $170. Please compute the sum and the mean of these prices. |
| 2 | For record ID 98765, use the sum from the previous calculation (800.0) to compute its absolute value and then find the square root of that sum. | For that record, use the calculated sum to compute its absolute value and then find the square root of that. |

**Changes:**
- "For record ID 98765" → "For that record"
- "use the sum from the previous calculation (800.0)" → "use the calculated sum"
- "find the square root of that sum" → "find the square root of that"

---

### Example 6: Twitter Engagement

| Turn | Before | After |
|------|--------|-------|
| 1 | tech_user/TechUser2024! authentication. Post tweet 'Our Q4 2024 roadmap is live!... | tech_user/TechUser2024! authentication. Post tweet 'Our Q4 2024 roadmap is live!... |
| 2 | Get the tweet we just posted and add a comment 'Thanks for the overwhelming response!... | Get that tweet and add a comment 'Thanks for the overwhelming response!... |

**Changes:** "the tweet we just posted" → "that tweet"

---

## Summary of Transformation Types

1. **Pronoun replacement**: "the tweet we just posted" → "that tweet"
2. **Value removal**: "(800.0)" removed, "ticket 654322" → "that ticket"
3. **Phrase shortening**: "Based on the current fuel level" → "Based on the fuel level"
4. **Indirect reference**: "the distance to San Francisco" → "the distance to that destination"
5. **Resolution text**: "'Restarted the server successfully'" → "the resolution details"

## Design Goals

The post-processing ensures that:
1. Explicit values from previous turns are not repeated
2. Queries reference the dialog history naturally
3. The model must track state/context to understand what values were computed
4. Credentials and auth info are preserved when needed for tool execution

## Files

- `src/anaphoric_processor.py` - The LLM-based post-processing script
- `data/anaphoric/quick_output_v3.jsonl` - Processed datapoints (examples 5-6)
- `data/anaphoric/quick_output.jsonl` - Original processed datapoints

## Usage

```bash
# Process datapoints
python src/anaphoric_processor.py input.jsonl output.jsonl

# Preview changes without applying
python src/anaphoric_processor.py input.jsonl output.jsonl --dry-run
```

## Processing Statistics

From 6 processed datapoints:
- 5 had Turn 2 queries transformed (83%)
- 1 had Turn 2 stay the same (17%) - Turn 2 already used anaphoric references
- Turn 1 queries generally not changed (no prior context to reference)