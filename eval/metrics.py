"""
Evaluation Metrics for NeuroMem v3.

Upgraded from substring matching to SEMANTIC SIMILARITY matching.
This fixes the fundamental problem where paraphrased memories
(e.g., "Works at TechFlow" vs "User is a software engineer at TechFlow")
were incorrectly scored as 0.0 by the old exact-match system.

Metrics:
- Recall@K: Does the expected fact appear in top-K (semantic match)?
- Memory Precision: Fraction of retrieved memories matching ground truth
- MRR: Average of 1/rank of first semantically relevant memory
- F1-Score: Harmonic mean of precision and recall
- Must-Not-Contain: Checks that invalidated facts are NOT present
- Latency percentiles: p50, p95, p99

Matching strategy:
  - Primary: Embedding cosine similarity > 0.60 (generous threshold
    that catches paraphrasing while rejecting unrelated content)
  - Fallback: Substring containment for exact terms like names/places
"""
import numpy as np
from typing import List, Optional


# ── Semantic Matching ────────────────────────────────────────────────

def semantic_match(
    candidate: str,
    expected: str,
    embedder=None,
    threshold: float = 0.60,
) -> bool:
    """
    Check if candidate semantically matches expected fact.

    Uses a two-pass strategy:
    1. Fast keyword check (names, places, specific terms)
    2. Embedding cosine similarity (catches paraphrasing)

    Args:
        candidate: Retrieved memory text
        expected: Expected ground truth text
        embedder: Optional Embedder instance for vector similarity
        threshold: Cosine similarity threshold (default 0.60)
    """
    candidate_lower = candidate.lower()
    expected_lower = expected.lower()

    # Pass 1: Direct keyword containment (fast path)
    if expected_lower in candidate_lower:
        return True

    # Pass 1b: Check individual expected words (for short terms like "sushi", "Austin")
    expected_words = expected_lower.split()
    if len(expected_words) <= 3:
        if all(word in candidate_lower for word in expected_words):
            return True

    # Pass 2: Embedding similarity (if embedder available)
    if embedder is not None:
        try:
            emb_candidate = np.array(embedder.embed(candidate))
            emb_expected = np.array(embedder.embed(expected))
            norm_c = np.linalg.norm(emb_candidate)
            norm_e = np.linalg.norm(emb_expected)
            if norm_c > 0 and norm_e > 0:
                similarity = float(np.dot(emb_candidate, emb_expected) / (norm_c * norm_e))
                return similarity > threshold
        except Exception:
            pass

    return False


def contains_forbidden(
    candidate: str, forbidden_terms: List[str]
) -> bool:
    """Check if a retrieved memory contains a forbidden/invalidated term."""
    candidate_lower = candidate.lower()
    for term in forbidden_terms:
        if term.lower() in candidate_lower:
            return True
    return False


# ── Recall Metrics ───────────────────────────────────────────────────

def recall_at_k(
    retrieved: List[str],
    expected_facts: List[str],
    k: int = 1,
    embedder=None,
) -> float:
    """
    Fraction of expected facts found in top-k retrieved memories.
    Uses semantic matching instead of exact substring.
    """
    if not expected_facts:
        return 1.0
    top_k = retrieved[:k]
    found = 0
    for expected in expected_facts:
        if any(semantic_match(r, expected, embedder) for r in top_k):
            found += 1
    return found / len(expected_facts)


def mean_reciprocal_rank(
    retrieved: List[str],
    expected_facts: List[str],
    embedder=None,
) -> float:
    """
    Mean Reciprocal Rank across all expected facts.
    MRR = mean(1/rank) for each expected fact's first match.
    """
    if not retrieved or not expected_facts:
        return 0.0
    
    reciprocal_ranks = []
    for expected in expected_facts:
        for i, item in enumerate(retrieved):
            if semantic_match(item, expected, embedder):
                reciprocal_ranks.append(1.0 / (i + 1))
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def memory_precision(
    retrieved: List[str],
    ground_truth: List[str],
    embedder=None,
) -> float:
    """Fraction of retrieved memories that match any ground truth fact."""
    if not retrieved:
        return 0.0
    hits = sum(
        1 for r in retrieved
        if any(semantic_match(r, gt, embedder) for gt in ground_truth)
    )
    return hits / len(retrieved)


def memory_f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


# ── Violation Detection ──────────────────────────────────────────────

def contradiction_leakage(
    retrieved: List[str],
    must_not_contain: List[str],
) -> dict:
    """
    Check if any invalidated facts leaked into the retrieval results.

    Returns:
        {
            "leaked": bool,
            "leaked_terms": [...],
            "clean_rate": 0.0-1.0
        }
    """
    if not must_not_contain:
        return {"leaked": False, "leaked_terms": [], "clean_rate": 1.0}

    leaked_terms = []
    for r in retrieved:
        for term in must_not_contain:
            if contains_forbidden(r, [term]):
                leaked_terms.append(term)

    return {
        "leaked": len(leaked_terms) > 0,
        "leaked_terms": list(set(leaked_terms)),
        "clean_rate": 1.0 - (len(set(leaked_terms)) / len(must_not_contain)),
    }


# ── Latency Stats ───────────────────────────────────────────────────

def compute_percentile(values: List[float], percentile: float) -> float:
    """Compute the Nth percentile of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * percentile)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def compute_p50(values: List[float]) -> float:
    return compute_percentile(values, 0.50)


def compute_p95(values: List[float]) -> float:
    return compute_percentile(values, 0.95)


def compute_p99(values: List[float]) -> float:
    return compute_percentile(values, 0.99)
