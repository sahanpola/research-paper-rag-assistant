import sys
from pathlib import Path
from app.config.settings import initialize_settings 
from app.ingestion.loader import load_documents_from_directory, build_index_with_metadata
from app.ingestion.chunker import create_chunks 


DATA_DIRECTORY = "./data/raw"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_VALUES = [3,5,10]
DEFAULT_QUERY = "What is the primary methodology or model used in this paper?"
SNIPPET_LENGTH = 180 


def get_test_query() -> str:
     """Get a retrieval evaluation question from the user."""
     query = input(
        "\nEnter a test question (press Enter for the default): "
    ).strip()

     return query or DEFAULT_QUERY 

def build_retriever_index() :
     """Load documents, create chunks, and build the vector index."""
     documents = load_documents_from_directory(DATA_DIRECTORY)

     if not documents:
        print(f"No documents found in '{DATA_DIRECTORY}'. Add a PDF first.")
        return None

     nodes = create_chunks(
         documents,
         chunk_size=CHUNK_SIZE,
         chunk_overlap=CHUNK_OVERLAP,
        )
     if not nodes:
         print("No chunks were generated from the documents.")
         return None

     return build_index_with_metadata(nodes)

def display_retrieved_chunks(
        retrieved_nodes,
        query: str,
        top_k: int,
) -> None:
    """Display retrieval results with scores and source metadata."""

    print("\n" + "=" * 50)
    print(f"RETRIEVAL RESULTS: top_k = {top_k}")
    print("=" * 50)
    print(f"Query: {query}")
    print(f"Retrieved {len(retrieved_nodes)} chunks:\n")

    for rank, result in enumerate(retrieved_nodes, start=1):
        score =(
            f"{result.score:.4f}"
            if result.score is not None
            else "N/A"
        )
        content =(
            result.node.get_content()
            .strip()
            .replace("\n"," ")

        )
        metadata = result.node.metadata
        page = metadata.get("page_label","N/A")
        file_name = metadata.get("file_name","Unknown")
        paper_title = metadata.get("paper_title", file_name)

        print(
            f"[{rank}] Score: {score} | "
            f"Paper: {paper_title} | Page: {page}"
        )
        print(f"    Snippet: {content[:SNIPPET_LENGTH]}...")
        print("-" * 40)


def run_retrieval_experiment() -> None:
    """Compare retrieval results across different top_k values."""
    print("=== Research Paper RAG — Sprint 4: Retrieval Experiments ===")

    initialize_settings()

    try:
        index = build_retriever_index()

    except Exception as error:
        print(f"Error] Failed to prepare retrieval index: {error}")
        return

    if index is None:
        return

    test_query = get_test_query()

    for top_k in TOP_K_VALUES:
        retriever = index.as_retriever(similarity_top=top_k)
        retrieved_nodes = retriever.retrieve(test_query)

        display_retrieved_chunks(
            retrieved_nodes,
            query=test_query,
            top_k=top_k,
        )

if __name__ == "__main__":
    run_retrieval_experiment()



