from typing import List 

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank 
from llama_index.core.retrievers import QueryFusionRetriever 
from llama_index.core.schema import NodeWithScore 

from app.config.settings import initialize_settings 
from app.ingestion.chunker import create_chunks 
from app.ingestion.loader import load_documents_from_directory 
from app.retrieval.hybrid_retriever import create_hybrid_retriever 

# Configuration 
CHUNK_SIZE = 512 
CHUNK_OVERLAP = 50
INITIAL_TOP_K = 15 
FINAL_TOP_K = 3 
RERANKER_MODEL = "BAAI/bge-reranker-base"

TEST_QUERY = "What is the primary methodology or model used in this paper?"

def get_reranked_nodes(
        retriever: QueryFusionRetriever,
        query: str,
        initial_top_k: int = INITIAL_TOP_K,
        final_top_k : int = FINAL_TOP_K,
        model_name: str = RERANKER_MODEL,
) -> List[NodeWithScore]: 
    """Retrieve candidate chunks and rerank them by query relevance."""

    print(f"[Reranker] Retrieving up to {initial_top_k} candidates...")

    # Retrieve a broad candidate pool.
    retriever.similarity_top_k = initial_top_k 
    candidate_nodes = retriever.retrieve(query)

    if not candidate_nodes :
        print("[Reranker] No candidate nodes found.")
        return []

    print(
        f"[Reranker] Reranking candidates with {model_name}..."
    )

    reranker = SentenceTransformerRerank(
        model=model_name,
        top_n=final_top_k,
    )

    reranked_nodes = reranker.postprocess_nodes(
        candidate_nodes,
        query_str=query,
    )

    print(
        f"[Reranker] Returned {len(reranked_nodes)} reranked nodes."
    )

    return reranked_nodes 

def run_reranking_experiment() -> None:
    """Build the retrieval pipeline and evaluate reranked results."""

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
        index = index,
        nodes = nodes ,
        similarity_top_k=INITIAL_TOP_K,
    )
    results = get_reranked_nodes(
        retriever=hybrid_retriever,
        query=TEST_QUERY,
        initial_top_k=INITIAL_TOP_K,
        final_top_k=FINAL_TOP_K,
    )

    print(f"\nQuery: {TEST_QUERY}")
    print(f"Final results: {len(results)}")

    for rank, result in enumerate(results, start=1):
        score = result.score
        score_text = f"{score:.4f}" if score is not None else "N/A"
        content = result.node.get_content().strip()

        print(f"\n--- Result {rank} | Score: {score_text} ---")
        print(f"{content[:200]}...")


if __name__ == "__main__":
    run_reranking_experiment()


