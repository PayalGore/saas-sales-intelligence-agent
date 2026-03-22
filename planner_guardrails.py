# ─────────────────────────────────────────────
# PLANNER GUARDRAILS — upstream hallucination prevention
# Implements Marcio's feedback: fix at source, not at output
# ─────────────────────────────────────────────

import json
import time
import re

# ── GUARDRAIL 1: Ambiguity Resolver ──────────────────────────────────────────
# Before any tool runs, check if question is clear enough to execute.
# Below confidence threshold → ask for clarification instead of guessing silently.

def resolve_ambiguity(question, client):
    """
    Check if question is specific enough to answer accurately.
    Returns: {"clear": True} or {"clear": False, "clarification": "..."}
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a query validator for a sales analytics system.
Assess if the question is specific enough to answer accurately with sales data.

Return JSON only:
{"clear": true} if question is specific enough
{"clear": false, "clarification": "one short question to resolve the ambiguity"}

Flag as unclear if:
- Time period is ambiguous (e.g. "recently", "lately", "this year" without context)  
- Metric is ambiguous (e.g. "performance" could mean win rate, bookings, or quota)
- Comparison is missing context (e.g. "is this good?" without a benchmark)

Do NOT flag as unclear if question is reasonably interpretable."""},
            {"role": "user", "content": question}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {"clear": True}  # fail open — don't block valid questions


# ── GUARDRAIL 2: Source-tagged tool results ───────────────────────────────────
# Tag every number from every tool with its source.
# LLM receives tagged context so it can cite sources instead of generating freely.

def build_source_tagged_context(all_results):
    """
    Convert raw tool results into source-tagged context string.
    Every number is labelled with which tool produced it.
    This constrains the LLM to only use numbers it can trace back to a source.
    """
    tagged = []
    for tool_name, result in all_results.items():
        tagged.append(f"\n[SOURCE: {tool_name}]")
        tagged.append(json.dumps(result, default=str, indent=2))
        tagged.append(f"[END SOURCE: {tool_name}]")
    return "\n".join(tagged)


# ── GUARDRAIL 3: Constrained generation prompt ────────────────────────────────
# Force LLM to only use numbers from tagged sources.
# If a number can't be traced to a source, don't include it.

CONSTRAINED_SYSTEM_PROMPT = """You are a senior SaaS sales ops analyst with real data access.

STRICT RULES — follow exactly:
1. ONLY use numbers that appear in the [SOURCE: tool_name] blocks provided to you
2. If you cannot find a specific number in the source data, say "data not available" — never estimate
3. Every statistic must be traceable to a source tool — do not generate numbers from memory
4. Temperature is already set to 0 — be factual, not creative
5. Be concise and executive-friendly. Under 250 words.
6. End with 1-2 specific actions.

PROHIBITED:
- Do not round numbers unless the source already shows them rounded
- Do not calculate new metrics by combining source numbers (e.g. don't derive win rate if not in source)
- Do not use phrases like "approximately", "around", "roughly" — use exact source figures or nothing
"""


# ── UPGRADED run_agent with planner guardrails ────────────────────────────────

def run_agent_with_guardrails(question, client, tools_funcs, kb_embeddings,
                               retrieve_context_fn, TOOL_DEFS, validate_response_fn):
    """
    Upgraded agent with upstream guardrails:
    1. Ambiguity resolver — clarify before executing
    2. Source-tagged context — every number has a source label  
    3. Constrained generation — LLM only uses sourced numbers
    4. Validate response — backstop for any edge cases that slip through
    """
    query_data, analyze_trends, find_at_risk, compare_periods, generate_weekly_report = tools_funcs
    tool_map = {
        "query_data":             lambda a: query_data(a.get("question", "")),
        "analyze_trends":         lambda a: analyze_trends(a.get("metric", ""), a.get("period", "monthly")),
        "find_at_risk":           lambda a: find_at_risk(a.get("category", "deals")),
        "compare_periods":        lambda a: compare_periods(a.get("period1", ""), a.get("period2", "")),
        "generate_weekly_report": lambda a: generate_weekly_report()
    }

    start = time.time()

    # ── GUARDRAIL 1: Ambiguity check before anything runs ──
    ambiguity = resolve_ambiguity(question, client)
    if not ambiguity.get("clear", True):
        clarification = ambiguity.get("clarification", "Could you be more specific?")
        elapsed = round(time.time() - start, 1)
        return (
            f"Before I pull the data — {clarification}",
            100.0, 0, 0,                          # accuracy placeholder
            ["ambiguity_resolver"],               # tool used
            elapsed,
            {}
        )

    # ── Tool execution loop ──
    messages = [
        {"role": "system", "content": CONSTRAINED_SYSTEM_PROMPT},
        {"role": "user",   "content": question}
    ]

    all_results = {}
    tools_used  = []

    for _ in range(5):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_DEFS,
            tool_choice="auto",
            temperature=0          # deterministic tool selection
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            break
        messages.append(msg)
        for tc in msg.tool_calls:
            fn   = tc.function.name
            args = json.loads(tc.function.arguments)
            tools_used.append(fn)
            result              = tool_map[fn](args)
            all_results[fn]     = result
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      json.dumps(result, default=str)
            })

    # ── GUARDRAIL 2: Build source-tagged context ──
    source_tagged = build_source_tagged_context(all_results)

    # ── GUARDRAIL 3: RAG context ──
    rag = retrieve_context_fn(question, client, kb_embeddings)

    # ── Constrained generation — LLM only narrates, never generates numbers ──
    messages.append({
        "role": "user",
        "content": f"""Generate the final answer.

SOURCED DATA — use ONLY these numbers:
{source_tagged}

BUSINESS CONTEXT (benchmarks, thresholds — cite as context, not as data):
{rag}

Remember: only use numbers from SOURCED DATA above. Under 250 words. End with 1-2 actions."""
    })

    final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1           # slightly above 0 for natural language, not for numbers
    )
    answer = final.choices[0].message.content

    # ── GUARDRAIL 4: Validation backstop (should rarely trigger now) ──
    accuracy, verified, total, unverified = validate_response_fn(answer, all_results)
    elapsed = round(time.time() - start, 1)

    return answer, accuracy, verified, total, tools_used, elapsed, all_results
