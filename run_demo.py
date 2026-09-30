"""run_demo.py -- end-to-end RAG Q&A through the deployed routewise router.

Shows: difficulty-based tier routing (scored on the bare question, context sent
via messages), semantic cache hits on repeat questions, the web-search branch,
and a forced-tier cost vs quality comparison (bypassing the cache so each tier
really runs).
"""
import time

import pandas as pd

import config
from answer import Index, ask

QUESTIONS = [
    "What is a transformer architecture?",
    "What does retrieval-augmented generation do?",
    "What is fine-tuning in machine learning?",
    "What causes hallucinations in language models?",
    "How do attention mechanisms and recurrent networks differ for language modeling?",
]
TIMELY = "What is the latest news about large language models this month?"
FORCE_TIERS = ["cheap", "mid", "frontier"]


def row(result: dict) -> dict:
    cost = result.get("cost_usd", 0.0)
    return {
        "question": result["question"][:45],
        "tier": result.get("routed_to", "?"),
        "difficulty": (
            f"{result['difficulty_score']:.2f}"
            if result.get("difficulty_score") is not None
            else "-"
        ),
        "cost_usd": f"${cost:.5f}",
        "cache_hit": result.get("cache_hit", False),
        "fallback": result.get("fallback_used", False),
        "latency_ms": round(result.get("latency_ms", 0), 0),
        "reason": result.get("route_reason", "")[:60],
    }


def main() -> int:
    index = Index()
    print(f"Router: {config.base_url()}")
    print("=" * 100)

    rows = []
    for q in QUESTIONS:
        print(f"Q: {q}")
        result = ask(index, q)
        rows.append(row(result))
        print(f"   -> {result.get('routed_to')} | ${result.get('cost_usd', 0):.5f} | "
              f"{result.get('latency_ms', 0):.0f}ms | {result.get('route_reason', '')}")

    print("\n--- Repeat question (semantic cache proof) ---")
    first = ask(index, QUESTIONS[0])
    time.sleep(0.5)
    again = ask(index, QUESTIONS[0])
    first["question"] = again["question"] = QUESTIONS[0] + " [repeat]"
    rows.append(row(again))
    print(f"   1st: cache={first.get('cache_hit')} cost=${first.get('cost_usd', 0):.5f}")
    print(f"   2nd: cache={again.get('cache_hit')} cost=${again.get('cost_usd', 0):.5f} "
          f"saved=${again.get('tokens_saved_usd', 0):.5f}")

    print("\n--- Time-sensitive question (web-search branch) ---")
    timely = ask(index, TIMELY)
    timely["question"] = TIMELY
    rows.append(row(timely))
    print(f"   -> routed_to={timely.get('routed_to')} | model={timely.get('model_id')} "
          f"| cost=${timely.get('cost_usd', 0):.5f}")

    print("\n--- Same question forced through each tier (cache bypassed) ---")
    forced_rows = []
    for tier in FORCE_TIERS:
        r = ask(index, QUESTIONS[4], override_tier=tier, bypass_cache=True)
        r["question"] = f"{QUESTIONS[4]} [force {tier}]"
        forced_rows.append(row(r))
        print(f"   force {tier:8s} -> ${r.get('cost_usd', 0):.5f} | "
              f"{r.get('latency_ms', 0):.0f}ms | {(r.get('response') or '')[:70]}")

    print("\n" + "=" * 100)
    print("AUTO-ROUTED RAG SESSION")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nFORCED-TIER COMPARISON (same question, cache bypassed)")
    print(pd.DataFrame(forced_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
