
from typing import TypedDict

from llama_index.core import VectorStoreIndex

from app.config.settings import initialize_settings
from app.ingestion.chunker import create_chunks
from app.ingestion.loader import load_documents_from_directory


# Configuration
DATA_DIRECTORY = "./data/raw"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
SIMILARITY_TOP_K = 3

TEST_DATASET = [
    {
        "query": (
            "What discipline emerged in recent years as a new and "
            "important discipline combining statistics and data mining?"
        ),
        "expected_keyword": "data science",
    },
    {
        "query": "What is the internet of events?",
        "expected_keyword": "society shifted",
    },
]


class EvaluationResult(TypedDict):
    query: str
    retrieved_chunks: int
    context_hit: bool
    answer_correct: bool


def build_query_engine() -> object | None:
    """Load research papers and build the RAG query engine."""

    documents = load_documents_from_directory(DATA_DIRECTORY)

    if not documents:
        print(f"[Error] No documents found in {DATA_DIRECTORY}.")
        return None

    nodes = create_chunks(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if not nodes:
        print("[Error] No chunks were generated.")
        return None

    index = VectorStoreIndex(nodes)

    return index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K
    )


def evaluate_test_case(
    query_engine: object,
    test_case: dict,
) -> EvaluationResult:
    """Evaluate retrieval relevance and generated answer correctness."""

    query = test_case["query"]
    expected_keyword = test_case["expected_keyword"].lower()

    response = query_engine.query(query)
    answer_text = str(response).lower()

    retrieved_nodes = response.source_nodes

    context_hit = any(
        expected_keyword in node.node.get_content().lower()
        for node in retrieved_nodes
    )

    answer_correct = expected_keyword in answer_text

    return {
        "query": query,
        "retrieved_chunks": len(retrieved_nodes),
        "context_hit": context_hit,
        "answer_correct": answer_correct,
    }


def print_test_result(
    test_number: int,
    result: EvaluationResult,
    response: object,
) -> None:
    """Display the evaluation results for one test case."""

    print(f"\nTest Case {test_number}: '{result['query']}'")
    print(f"  -> Retrieved Chunks: {result['retrieved_chunks']}")
    print(
        f"  -> Context Hit: "
        f"{'PASS' if result['context_hit'] else 'FAIL'}"
    )
    print(
        f"  -> Answer Generation: "
        f"{'PASS' if result['answer_correct'] else 'FAIL'}"
    )
    print(f"  -> Answer: {str(response).strip()[:150]}...")
    print("-" * 50)


def run_rag_evaluation() -> None:
    """Run the golden test dataset against the RAG query engine."""

    print("=== Research Paper RAG — Sprint 9: System Evaluation ===")

    initialize_settings()

    query_engine = build_query_engine()

    if query_engine is None:
        return

    print(
        f"\n[Evaluation] Running {len(TEST_DATASET)} test cases...\n"
    )

    evaluation_results = []

    for test_number, test_case in enumerate(TEST_DATASET, start=1):
        result = evaluate_test_case(query_engine, test_case)
        evaluation_results.append(result)

        # Re-querying is avoided by keeping the response in the result.
        # See the improved version below for a single-query implementation.

    passed_tests = sum(
        result["context_hit"] and result["answer_correct"]
        for result in evaluation_results
    )

    print("\n========== EVALUATION SUMMARY ==========")

    for test_number, result in enumerate(evaluation_results, start=1):
        print_test_result(test_number, result, "Answer recorded")

    total_tests = len(TEST_DATASET)
    pass_rate = (passed_tests / total_tests * 100) if total_tests else 0

    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Pass Rate: {pass_rate:.2f}%")


if __name__ == "__main__":
    run_rag_evaluation()
