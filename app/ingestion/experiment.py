from llama_index.core import VectorStoreIndex 
from llama_index.core.schema import TextNode 

from app.config.settings import initialize_settings
from app.ingestion.chunker import create_chunks
from app.ingestion.loader import load_documents_from_directory 

DATA_DIRECTORY = "./data/raw"
TEST_QUERY ="What is the main methodology or model used in this paper"

CHUNK_SIZES = [256,512,1024]
CHUNK_OVERLAP_RATIO = 0.1
TOP_K_RESULTS = 3
PREVIEW_LENGTH = 300

def retrieve_relavent_chunks(
        nodes: list[TextNode],
        query: str,
) -> list:
    """ Retrieve the most relevant chunks for a query. """
    index = VectorStoreIndex(nodes)
    retriever = index.as_retriever(
        similarity_top_k=TOP_K_RESULTS,
    )

    return retriever.retrieve(query)

def display_retrieved_chunks(retrieved_nodes,start=1):
    """ Display retrieved chunks and their similarity scores."""
    if not retrieved_nodes:
        print("No relevant chunks were retrieved.")
        return
    for rank, result in enumerate(retrieved_nodes, start=1):
        score = (
            f"{result.score:.4f}"
            if result.score is not None
            else "N/A"
        )
        content_preview = result.node.get_content()[:PREVIEW_LENGTH]
        print(f"\n--- Result {rank} (Score: {score}) ---")
        print(f"{content_preview}...")

def run_checking_experiment() -> None:
    """Compare retrieval results across different chunk sizes."""
    initialize_settings()
    documents = load_documents_from_directory(DATA_DIRECTORY)

    if not documents:
        print(
            f"No documents found in '{DATA_DIRECTORY}' ."
            "Add a PDF first!"
        )
        return 
    for chunk_size in CHUNK_SIZES :
        print(
            f"\n{'=' * 16} "
            f"CHUNK SIZE: {chunk_size} " 
            f"{'=' * 16}"

        )

        chunk_overlap = int(chunk_size*CHUNK_OVERLAP_RATIO)

        nodes = create_chunks(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        retrieved_nodes = retrieve_relavent_chunks(
            nodes,
            TEST_QUERY,
        )

        display_retrieved_chunks(retrieved_nodes)

if __name__ == "__main__":
    run_checking_experiment()
    




