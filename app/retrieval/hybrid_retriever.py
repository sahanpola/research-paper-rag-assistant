
from typing import Sequence

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from app.config.settings import initialize_settings
from app.ingestion.chunker import create_chunks
from app.ingestion.loader import load_documents_from_directory


# Configuration constants
DEFAULT_TOP_K = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TEST_QUERY = "What is data science?"


def create_hybrid_retriever(
    index: VectorStoreIndex,
    nodes: Sequence[Document],
    similarity_top_k: int = DEFAULT_TOP_K,
) -> QueryFusionRetriever:
    """Combine dense vector search and sparse BM25 search using rank fusion."""

    print("[Retriever] Initializing hybrid retriever...")

    vector_retriever = index.as_retriever(
        similarity_top_k=similarity_top_k
    )

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=list(nodes),
        similarity_top_k=similarity_top_k,
    )

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=similarity_top_k,
        num_queries=1,
        use_async=False,
        mode="reciprocal_rerank",
    )

    print("[Retriever] Hybrid retriever initialized successfully.")

    return hybrid_retriever


def run_hybrid_retrieval_experiment() -> None:
    """Load research papers, build an index, and test hybrid retrieval."""

    initialize_settings()

    documents = load_documents_from_directory("./data/raw")

    if not documents:
        print("[Error] No documents found in ./data/raw.")
        return

    nodes = create_chunks(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if not nodes:
        print("[Error] No chunks were created.")
        return

    index = VectorStoreIndex(nodes)

    hybrid_retriever = create_hybrid_retriever(
        index=index,
        nodes=nodes,
        similarity_top_k=3,
    )

    results = hybrid_retriever.retrieve(TEST_QUERY)

    print(f"\nQuery: {TEST_QUERY}")
    print(f"Retrieved {len(results)} results:\n")

    for rank, result in enumerate(results, start=1):
        content = result.node.get_content().strip()
        metadata = result.node.metadata

        print(
            f"[{rank}] Score: {result.score if result.score is not None else 'N/A'}"
        )
        print(f"Paper: {metadata.get('paper_title', 'Unknown')}")
        print(f"Content: {content[:200]}...")
        print("-" * 50)


if __name__ == "__main__":
    run_hybrid_retrieval_experiment()
