
import math


def _unique_items(items: list[str]) -> list[str]:
    """Remove duplicates while preserving retrieval order."""

    return list(dict.fromkeys(items))


def _validate_k(k: int) -> None:
    """Ensure K is a positive integer."""

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer.")


def calculate_hit_rate(
    retrieved_pages: list[str],
    ground_truth_pages: list[str],
) -> float:
    """Return 1 if at least one relevant page was retrieved."""

    if not ground_truth_pages:
        return 0.0

    relevant_pages = set(ground_truth_pages)

    return float(
        any(page in relevant_pages for page in retrieved_pages)
    )


def calculate_precision_at_k(
    retrieved_pages: list[str],
    ground_truth_pages: list[str],
    k: int,
) -> float:
    """Calculate Precision@K using unique retrieved pages."""

    _validate_k(k)

    if not ground_truth_pages:
        return 0.0

    top_k = _unique_items(retrieved_pages)[:k]
    relevant_pages = set(ground_truth_pages)

    relevant_count = sum(
        page in relevant_pages for page in top_k
    )

    return relevant_count / k


def calculate_recall_at_k(
    retrieved_pages: list[str],
    ground_truth_pages: list[str],
    k: int,
) -> float:
    """Calculate Recall@K using unique relevant pages."""

    _validate_k(k)

    if not ground_truth_pages:
        return 0.0

    top_k = _unique_items(retrieved_pages)[:k]
    relevant_pages = set(ground_truth_pages)

    relevant_count = sum(
        page in relevant_pages for page in top_k
    )

    return relevant_count / len(relevant_pages)


def calculate_mrr(
    retrieved_pages: list[str],
    ground_truth_pages: list[str],
) -> float:
    """Calculate Mean Reciprocal Rank for a single query."""

    if not ground_truth_pages:
        return 0.0

    relevant_pages = set(ground_truth_pages)
    unique_retrieved = _unique_items(retrieved_pages)

    for rank, page in enumerate(unique_retrieved, start=1):
        if page in relevant_pages:
            return 1.0 / rank

    return 0.0


def calculate_ndcg_at_k(
    retrieved_pages: list[str],
    ground_truth_pages: list[str],
    k: int,
) -> float:
    """Calculate binary-relevance NDCG@K."""

    _validate_k(k)

    if not ground_truth_pages:
        return 0.0

    top_k = _unique_items(retrieved_pages)[:k]
    relevant_pages = set(ground_truth_pages)

    # Calculate DCG
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, page in enumerate(top_k, start=1)
        if page in relevant_pages
    )

    # Calculate ideal DCG
    ideal_relevant_count = min(len(relevant_pages), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


if __name__ == "__main__":

    print("=== RAG Evaluation Metrics: Sanity Tests ===")

    retrieved = ["1", "3", "5"]
    ground_truth = ["1", "2"]

    print("\nTest 1: Normal retrieval")

    print("Hit Rate:", calculate_hit_rate(retrieved, ground_truth))
    print("Precision@3:", calculate_precision_at_k(retrieved, ground_truth, 3))
    print("Recall@3:", calculate_recall_at_k(retrieved, ground_truth, 3))
    print("MRR:", calculate_mrr(retrieved, ground_truth))
    print("NDCG@3:", calculate_ndcg_at_k(retrieved, ground_truth, 3))

    print("\nTest 2: Duplicate retrieval")

    duplicate_retrieved = ["1", "1", "2"]

    print(
        "Precision@3:",
        calculate_precision_at_k(duplicate_retrieved, ground_truth, 3),
    )
    print(
        "Recall@3:",
        calculate_recall_at_k(duplicate_retrieved, ground_truth, 3),
    )
    print(
        "NDCG@3:",
        calculate_ndcg_at_k(duplicate_retrieved, ground_truth, 3),
    )
