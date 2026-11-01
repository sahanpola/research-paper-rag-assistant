from llama_index.core import VectorStoreIndex 
from llama_index.core.retrievers import QueryFusionRetriever 
from llama_index.core.schema import NodeWithScore 

from app.config.settings import initialize_settings 
from app.ingestion.chunker import create_chunks
from app.ingestion.loader import load_documents_from_directory 

# Configuration 
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

DEFAULT_TOP_K= 5
DEFAULT_NUM_QUERIES = 3

TEST_QUERY = "How does data science work?"

def create_multi_query_retriever(
        index: VectorStoreIndex,
        similarity_top_k: int = DEFAULT_TOP_K,
        num_queries: int = DEFAULT_NUM_QUERIES,
) -> QueryFusionRetriever:
    """Create a retriever that expands queries and fuses retrieval results."""

    print(
        f"[QueryTransformer] Initializing multi-query retriever" 
        f"(num_queries={num_queries})..."
    )

    base_retriever = index.as_retriever(
        similarity_top_k=similarity_top_k
    )

    multi_query_retriever = QueryFusionRetriever(
        retrievers= [base_retriever],
        similarity_top_k=similarity_top_k,
        num_queries=num_queries,
        use_async=False,
        mode="reciprocal_rerank",

    )

    print("[QueryTransformer] Multi-query retriever initialized.")

    return multi_query_retriever

def run_multi_query_experiment() -> None:
    """Build the retrieval pipeline and test multi-query expansion."""

    initialize_settings(),
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

    retriever = create_multi_query_retriever(
        index=index,
        similarity_top_k=DEFAULT_TOP_K,
        num_queries=DEFAULT_NUM_QUERIES,
    )
    results = retriever.retrieve(TEST_QUERY)

    print(f"\nQuery: {TEST_QUERY}")
    print(f"Retrieved {len(results)} results using multi-query retrieval.\n")

    for rank, result in enumerate(results, start=1):
        content = result.node.get_content().strip()
        score = result.score
        score_text = f"{score:.4f}" if score is not None else "N/A"

        print(f"[{rank}] Score: {score_text}")
        print(f"Content: {content[:200]}...")
        print("-" * 50)


if __name__ == "__main__":
    run_multi_query_experiment()
