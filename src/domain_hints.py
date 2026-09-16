"""Domain-specific hints for API generation by category.

Keys are focus_category names matching BFCL categories.
Empty strings for domains that don't need special rules yet.
"""

DOMAIN_HINTS = {
    "Vehicle Control": """
=== VEHICLE CONTROL DOMAIN RULES ===
When generating queries for Vehicle Control:

1. TRIP FEASIBILITY QUERIES:
   - If user asks "can I make it to [destination]?" or "is the trip feasible?":
     - The distance MUST come from either:
       a) User explicitly states it: "can I make the 380 mile trip to Grand Canyon?"
       b) estimate_distance is called FIRST to compute the actual distance
     - Never use arbitrary distances like 100, 200, 300 unless user specifies
   - Do NOT call estimate_drive_feasibility_by_mileage alone without knowing actual distance

2. NAVIGATION SETUP:
   - When setting navigation destination, include a REAL distance via estimate_distance
   - Example: "Navigate to Grand Canyon (about 600 miles from here) and check if I have enough fuel"
   - expected_tools: [estimate_distance, estimate_drive_feasibility_by_mileage] OR [set_navigation, estimate_distance, estimate_drive_feasibility_by_mileage]

3. FUEL-RELATED QUERIES:
   - When user asks to check fuel level, displayCarStatus(option="fuel")
   - When user asks to add fuel, use fillFuelTank with fuelAmount
   - Ensure requests match tool capabilities (displayCarStatus shows ONE option at a time)

4. DISPLAY TOOLS:
   - displayCarStatus can only show ONE option per call (fuel, battery, doors, climate, etc.)
   - If user asks for multiple statuses, generate multiple tool calls
   - Example: "Check fuel and battery" -> [displayCarStatus(option="fuel"), displayCarStatus(option="battery")]

5. PREREQUISITE TOOLS:
   - Do NOT add prerequisite tools unless user explicitly mentions them
   - Example: "Start the engine" -> startEngine only, NOT pressBrakePedal + startEngine

6. STATE-CHANGING CONTROLS:
   - Compare every requested setting with generator state. Call startEngine only
     while stopped, lockDoors only while at least one door is unlocked, and
     setHeadlights/setCruiseControl only when the requested value differs.
   - If a later turn needs another mutation, request a genuinely different
     setting or an explicit reversible transition; never repeat a no-op merely
     to fill the exact call schedule.
""",
    "Travel Booking": """
=== TRAVEL BOOKING DOMAIN RULES ===
Authenticate before protected booking/card operations. A booking, invoice,
insurance purchase, or cancellation must consume an actual earlier booking or
card identifier where its schema requires one. Never invent those identifiers.
Do not register an already registered card or cancel a missing booking.
""",
    "Finance": """
=== FINANCE DOMAIN RULES ===
Authenticate before protected trading operations. Obtain symbols through the
available name-to-symbol lookup when the user supplies a company name. Do not
cancel a nonexistent order, repeat existing watchlist membership, or make an
update that leaves account/market state unchanged.
""",
    "Communication": """
=== COMMUNICATION DOMAIN RULES ===
Authenticate before protected messaging operations. Use list_users or
get_user_id to resolve an existing recipient before send_message. Do NOT call
add_contact merely as a prerequisite to messaging; use it only when the user
explicitly asks to add a genuinely new contact and the state makes that
mutation feasible. Never add an existing contact or send to an invented ID.
delete_message always needs one concrete message_id: bind it to an earlier
tool output that actually declares that exact field, or state the plausible ID
in the user request. "Latest message" is not enough when no selected call
returns its message_id.
""",
    "Science": """
=== SCIENCE DOMAIN RULES ===
When generating executable unit conversions:

1. imperial_si_conversion supports Fahrenheit↔Celsius and these reversible
   pairs (including square/cubic forms where meaningful): inch↔cm,
   pound↔kg, mile↔km, gallon↔liter, foot↔meter, yard↔meter, ounce↔gram.
2. si_unit_conversion requires the same base unit on both sides, for example
   meter↔kilometer, gram↔kilogram, liter↔milliliter, or byte↔megabyte.
3. Statistical tools that accept `numbers` require an actual numeric array;
   never bind a prior scalar `result` to an array parameter.
""",
    "Storage": """
=== STORAGE DOMAIN RULES ===
Every source path must exist when read, copied, moved, or deleted. Create a new
path first when the conversation needs one. Delete a directory only after its
contents are removed, and never repeat a mutation that the current state
already satisfies.
""",
    "Events": """
=== EVENTS DOMAIN RULES ===
Authenticate before protected ticket operations. Create or retrieve a real
ticket before editing, resolving, or closing it, and bind the later ticket ID
to that visible tool output. Never resolve/close a ticket already in that
state, and never invent a ticket ID. In particular, keep using `create_ticket`'s
returned `id` for "that ticket" or "the ticket I just created"; do not replace
it with whatever ticket a broad `get_user_tickets` read happens to return.
Despite its plural name, get_user_tickets has a single-object output schema and
cannot satisfy requests to list, enumerate, count, or summarize all tickets.
Phrase the request around the one matching ticket it returns. For edit_ticket,
every field included in `updates` must differ from current state; do not claim
to raise a priority to the value it already has while changing another field.
""",
    "Posting Api": """
=== POSTING DOMAIN RULES ===
Authenticate before protected posting operations. Bind comments, mentions,
retweets, and follow-up reads to a real earlier tweet/user result. Follow or
unfollow only when the generator state shows that the relationship will
actually change. A comment, mention, or retweet must also create a genuinely
new interaction in simulator state; do not repeat an existing interaction or
reuse the same mutation later merely to fill the call schedule.
""",
}


def get_domain_hints(focus_category: str) -> str:
    """Get hints for a specific domain, or empty string if none defined."""
    return DOMAIN_HINTS.get(focus_category, "")
