"""
NeuroMem v3 Evaluation Harness.

Adversarial evaluation that tests the 3 core capabilities that
differentiate NeuroMem from standard RAG:
  1. Contradiction Resolution — old facts must be invalidated
  2. Temporal Validity — only current state should be active
  3. Noise Filtering — surprise gate must block small talk

Uses SEMANTIC MATCHING (embedding similarity) instead of substring
matching to correctly evaluate paraphrased memories.

Generates:
  - Terminal report with pass/fail per test
  - JSON report at eval/results/report.json
  - Visualization charts via eval/visualize.py
"""
import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import httpx

from eval.metrics import (
    recall_at_k,
    mean_reciprocal_rank,
    memory_precision,
    memory_f1_score,
    contradiction_leakage,
    compute_p50,
    compute_p95,
    compute_p99,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CONVERSATIONS_DIR = Path(__file__).parent / "conversations"
RESULTS_DIR = Path(__file__).parent / "results"


class NeuroMemEvalHarness:
    """
    End-to-end evaluation harness for NeuroMem's cognitive pipeline.

    For each test case:
    1. Feeds the conversation turns through POST /chat
    2. Queries memories via GET /memory/retrieve
    3. Scores retrieval quality with semantic matching
    4. Checks for contradiction leakage (invalidated facts appearing)
    5. Records latency at every stage
    """

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": api_key,
            },
            timeout=httpx.Timeout(60.0),
        )
        self.results = []

    async def run_all(self) -> dict:
        """Run all test cases and generate aggregate report."""
        test_files = sorted(CONVERSATIONS_DIR.glob("tc_*.json"))
        if not test_files:
            logger.error("No test cases found in %s", CONVERSATIONS_DIR)
            return {}

        print(f"\n🧪 NeuroMem v3 Evaluation — {len(test_files)} adversarial tests\n")
        print("=" * 64)

        for test_file in test_files:
            result = await self.run_test_case(test_file)
            self.results.append(result)
            self._print_test_result(result)
            # Wait between tests to respect API rate limits
            await asyncio.sleep(15.0)

        report = self._generate_report()
        self._save_report(report)
        self._print_report(report)
        return report

    async def run_test_case(self, test_file: Path) -> dict:
        """Run a single test case end-to-end."""
        with open(test_file) as f:
            tc = json.load(f)

        result = {
            "name": tc["name"],
            "category": tc["category"],
            "file": test_file.name,
            "conversation_turns": len(tc["conversation"]) // 2,
            "queries": [],
            "latencies": {
                "ingest_ms": [],
                "retrieve_ms": [],
            },
            "overall_pass": True,
        }

        # ── Phase 1: Clear previous memories ─────────────────────────
        try:
            await self.client.delete("/memory/clear")
        except Exception:
            pass  # OK if no memories to clear

        # ── Phase 2: Ingest conversation turns ───────────────────────
        session_id = None
        for i in range(0, len(tc["conversation"]), 2):
            user_turn = tc["conversation"][i]
            if user_turn["role"] != "user":
                continue

            t0 = time.time()
            try:
                payload = {
                    "message": user_turn["content"],
                    "include_memory_debug": True,
                }
                if session_id:
                    payload["session_id"] = session_id

                resp = await self.client.post("/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                session_id = data.get("session_id", session_id)
                ingest_ms = int((time.time() - t0) * 1000)
                result["latencies"]["ingest_ms"].append(ingest_ms)

            except Exception as e:
                logger.warning(f"  ⚠️  Ingest failed for turn {i}: {e}")
                result["latencies"]["ingest_ms"].append(-1)

            # Rate limit protection between turns
            await asyncio.sleep(5.0)

        # ── Phase 3: Wait for background pipeline to finish ──────────
        await asyncio.sleep(8.0)

        # ── Phase 4: Run test queries ────────────────────────────────
        for tq in tc.get("test_queries", []):
            query_result = await self._run_query(tq)
            result["queries"].append(query_result)
            if not query_result["pass"]:
                result["overall_pass"] = False
            await asyncio.sleep(5.0)

        return result

    async def _run_query(self, test_query: dict) -> dict:
        """Execute a single test query and score the results."""
        query = test_query["query"]
        expected_facts = test_query["expected_facts"]
        must_not_contain = test_query.get("must_not_contain", [])

        t0 = time.time()
        try:
            resp = await self.client.get(
                "/memory/retrieve",
                params={"query": query},
            )
            resp.raise_for_status()
            data = resp.json()
            retrieve_ms = int((time.time() - t0) * 1000)
        except Exception as e:
            return {
                "query": query,
                "pass": False,
                "error": str(e),
                "retrieve_ms": -1,
            }

        # Extract memory contents from response
        retrieved = []
        if isinstance(data, list):
            retrieved = [item.get("content", "") for item in data]
        elif isinstance(data, dict):
            memories = data.get("memories", data.get("results", []))
            if isinstance(memories, list):
                retrieved = [
                    m.get("content", m) if isinstance(m, dict) else str(m)
                    for m in memories
                ]

        # ── Score with semantic matching ──────────────────────────────
        r_at_1 = recall_at_k(retrieved, expected_facts, k=1)
        r_at_5 = recall_at_k(retrieved, expected_facts, k=5)
        mrr = mean_reciprocal_rank(retrieved, expected_facts)
        precision = memory_precision(retrieved, expected_facts)
        f1 = memory_f1_score(precision, r_at_5)

        # ── Check contradiction leakage ──────────────────────────────
        leakage = contradiction_leakage(retrieved, must_not_contain)

        # A query passes if:
        # - At least one expected fact is found in top 5 (R@5 > 0)
        # - No invalidated facts leaked
        passed = r_at_5 > 0 and not leakage["leaked"]

        return {
            "query": query,
            "description": test_query.get("description", ""),
            "pass": passed,
            "metrics": {
                "recall_at_1": round(r_at_1, 3),
                "recall_at_5": round(r_at_5, 3),
                "mrr": round(mrr, 3),
                "precision": round(precision, 3),
                "f1": round(f1, 3),
            },
            "leakage": leakage,
            "retrieved_count": len(retrieved),
            "retrieved_preview": [r[:80] for r in retrieved[:5]],
            "retrieve_ms": retrieve_ms,
        }

    def _generate_report(self) -> dict:
        """Generate aggregate report across all test cases."""
        all_queries = []
        all_ingest_latencies = []
        all_retrieve_latencies = []
        per_category = {}

        for result in self.results:
            category = result["category"]
            if category not in per_category:
                per_category[category] = {"pass": 0, "fail": 0, "queries": []}

            for q in result["queries"]:
                all_queries.append(q)
                per_category[category]["queries"].append(q)
                if q["pass"]:
                    per_category[category]["pass"] += 1
                else:
                    per_category[category]["fail"] += 1

            all_ingest_latencies.extend(
                [l for l in result["latencies"]["ingest_ms"] if l > 0]
            )
            for q in result["queries"]:
                if q.get("retrieve_ms", -1) > 0:
                    all_retrieve_latencies.append(q["retrieve_ms"])

        # Aggregate metrics
        metrics = {m: [] for m in ["recall_at_1", "recall_at_5", "mrr", "precision", "f1"]}
        for q in all_queries:
            if "metrics" in q:
                for k, v in q["metrics"].items():
                    metrics[k].append(v)

        avg_metrics = {
            k: round(sum(v) / len(v), 3) if v else 0.0
            for k, v in metrics.items()
        }

        total_pass = sum(1 for q in all_queries if q.get("pass"))
        total_queries = len(all_queries)

        return {
            "timestamp": datetime.now().isoformat(),
            "version": "v3.0",
            "test_cases": len(self.results),
            "total_queries": total_queries,
            "passed": total_pass,
            "failed": total_queries - total_pass,
            "pass_rate": round(total_pass / total_queries, 3) if total_queries else 0,
            "aggregate_metrics": avg_metrics,
            "per_category": {
                cat: {
                    "pass_rate": round(
                        data["pass"] / (data["pass"] + data["fail"]), 3
                    ) if (data["pass"] + data["fail"]) > 0 else 0,
                    "passed": data["pass"],
                    "failed": data["fail"],
                }
                for cat, data in per_category.items()
            },
            "latency": {
                "ingest_p50_ms": round(compute_p50(all_ingest_latencies)),
                "ingest_p95_ms": round(compute_p95(all_ingest_latencies)),
                "retrieve_p50_ms": round(compute_p50(all_retrieve_latencies)),
                "retrieve_p95_ms": round(compute_p95(all_retrieve_latencies)),
                "retrieve_p99_ms": round(compute_p99(all_retrieve_latencies)),
            },
            "detailed_results": self.results,
        }

    def _print_test_result(self, result: dict):
        """Print a single test result to terminal."""
        status = "✅" if result["overall_pass"] else "❌"
        print(f"\n  {status} {result['name']} ({result['file']})")
        for q in result["queries"]:
            q_status = "✓" if q.get("pass") else "✗"
            metrics = q.get("metrics", {})
            print(
                f"    {q_status} {q['query'][:50]:50s} "
                f"R@1={metrics.get('recall_at_1', 0):.2f} "
                f"R@5={metrics.get('recall_at_5', 0):.2f} "
                f"F1={metrics.get('f1', 0):.2f}"
            )
            if q.get("leakage", {}).get("leaked"):
                print(f"      ⚠️  LEAKED: {q['leakage']['leaked_terms']}")

    def _print_report(self, report: dict):
        """Print the aggregate report to terminal."""
        print("\n" + "=" * 64)
        print("📊 NEUROMEM v3 EVALUATION REPORT")
        print("=" * 64)
        print(f"  Tests:          {report['test_cases']}")
        print(f"  Queries:        {report['total_queries']}")
        print(f"  Passed:         {report['passed']}/{report['total_queries']}")
        print(f"  Pass Rate:      {report['pass_rate']*100:.1f}%")
        print()
        print("  ── Aggregate Metrics ──")
        am = report["aggregate_metrics"]
        print(f"  Recall@1:       {am.get('recall_at_1', 0)*100:.1f}%")
        print(f"  Recall@5:       {am.get('recall_at_5', 0)*100:.1f}%")
        print(f"  MRR:            {am.get('mrr', 0):.3f}")
        print(f"  Precision:      {am.get('precision', 0)*100:.1f}%")
        print(f"  F1:             {am.get('f1', 0)*100:.1f}%")
        print()
        print("  ── Per Category ──")
        for cat, data in report["per_category"].items():
            print(f"  {cat:20s} {data['pass_rate']*100:5.1f}% ({data['passed']}/{data['passed']+data['failed']})")
        print()
        print("  ── Latency ──")
        lat = report["latency"]
        print(f"  Ingest  p50: {lat['ingest_p50_ms']}ms  p95: {lat['ingest_p95_ms']}ms")
        print(f"  Retrieve p50: {lat['retrieve_p50_ms']}ms  p95: {lat['retrieve_p95_ms']}ms  p99: {lat['retrieve_p99_ms']}ms")
        print("=" * 64)

    def _save_report(self, report: dict):
        """Save report to JSON file."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = RESULTS_DIR / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  📁 Report saved to {report_path}")

    async def close(self):
        await self.client.aclose()


async def main():
    parser = argparse.ArgumentParser(description="NeuroMem v3 Evaluation Harness")
    parser.add_argument("--api-key", default="neuromem-dev-key-change-me")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--visualize", action="store_true", help="Generate charts after eval")
    args = parser.parse_args()

    harness = NeuroMemEvalHarness(
        base_url=args.base_url,
        api_key=args.api_key,
    )
    try:
        report = await harness.run_all()

        if args.visualize and report:
            try:
                from eval.visualize import generate_all_charts
                generate_all_charts(report)
            except ImportError:
                print("\n  ⚠️  matplotlib not installed. Skipping visualizations.")
    finally:
        await harness.close()


if __name__ == "__main__":
    asyncio.run(main())
