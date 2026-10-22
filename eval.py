"""Run the golden set through the graph and print a metrics table.

Usage: python eval.py [--delay SECONDS]   (needs live Neo4j + Gemini credentials in .env)
"""
import argparse
import asyncio
import json
import re
import time
from uuid import uuid4

from agent import app, MONEY_PATTERNS


def has_money(text):
    return any(re.search(p, text, re.IGNORECASE) for p in MONEY_PATTERNS)


async def run_question(question):
    inputs = {"question": question, "retry_count": 0, "compliance_status": "", "generation": "", "context": ""}
    config = {"configurable": {"thread_id": str(uuid4())}}  # fresh thread per question
    first_generation, final = None, {}
    start = time.time()
    # stream_mode="updates" yields {node_name: state_update}, letting us see the first generate output
    async for update in app.astream(inputs, config=config, stream_mode="updates"):
        for node, delta in update.items():
            if node == "generate" and first_generation is None:
                first_generation = delta["generation"]
            final.update(delta)
    return {
        "first_generation": first_generation or "",
        "answer": final.get("generation", ""),
        "status": final.get("compliance_status", ""),
        "retries": final.get("retry_count", 0),
        "latency": time.time() - start,
    }


async def main(delay):
    with open("golden_set.json", encoding="utf-8") as f:
        golden = json.load(f)

    rows = []
    for item in golden:
        try:
            r = await run_question(item["question"])
        except Exception as e:  # e.g. rate limit / bad credentials: record and continue
            print(f"ERROR on {item['question']!r}: {e}")
            continue
        r["question"] = item["question"]
        r["correct"] = any(e.lower() in r["answer"].lower() for e in item["expected"])
        r["first_violation"] = has_money(r["first_generation"])
        rows.append(r)
        await asyncio.sleep(delay)

    if not rows:
        print("No questions completed.")
        return

    print(f"\n{'Question':<55} {'Correct':<8} {'1stViol':<8} {'Final':<6} {'Retries':<8} {'Latency':<8}")
    print("-" * 97)
    for r in rows:
        print(f"{r['question'][:54]:<55} {str(r['correct']):<8} {str(r['first_violation']):<8} "
              f"{r['status']:<6} {r['retries']:<8} {r['latency']:.2f}s")

    n = len(rows)
    print("\nSummary")
    print(f"  Questions completed:            {n}/{len(golden)}")
    print(f"  Answer correctness:             {sum(r['correct'] for r in rows)}/{n}")
    print(f"  First generation violated rule: {sum(r['first_violation'] for r in rows)}/{n}")
    print(f"  Final compliance pass rate:     {sum(r['status'] == 'PASS' for r in rows)}/{n}")
    print(f"  Average retries:                {sum(r['retries'] for r in rows) / n:.2f}")
    print(f"  Average latency:                {sum(r['latency'] for r in rows) / n:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0, help="seconds to sleep between questions (free-tier rate limits)")
    asyncio.run(main(parser.parse_args().delay))
