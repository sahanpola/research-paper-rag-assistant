
from app.ingestion.chunker import create_chunks
from llama_index.core import VectorStoreIndex
from app.config.settings import initialize_settings
from app.ingestion.loader import load_documents_from_directory


DATA_DIRECTORY = "./data/raw"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3
EXIT_COMMANDS={"exit","quit"}


def create_query_engine():
    """ Load documents, create chunks, and build the query engine"""
    documents = load_documents_from_directory(DATA_DIRECTORY)

    if not documents:
        print(f"No PDF files found in '{DATA_DIRECTORY}'")
        return None

    nodes = create_chunks(
        documents,
        chunk_size= CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    if not nodes:
        print("No text chunk generated from the documents")
        return None

    index = VectorStoreIndex(nodes)
    return index.as_query_engine(similarity_top_k=TOP_K_RESULTS)

def display_response(response) -> None:
    """Display the answer and its source citations."""
    print("\n" + "-" * 20 + " ANSWER " + "-" * 20)
    print(str(response).strip())
    print("-" * 48)

    print("\n[Structured Source Citations]:")

    for rank, source in enumerate(response.source_nodes, start=1):
        metadata = source.node.metadata
        score = (
            f"{source.score:.4f}"
            if source.score is not None
            else "N/A"
        )

        file_name = metadata.get("file_name", "Unknown")
        page = metadata.get("page_label", "N/A")
        paper_title = metadata.get("paper_title", file_name)

        print(f"  [{rank}] Paper: {paper_title}")
        print(f"      File: {file_name} | Page: {page} | Score: {score}")


def run_query_loop(query_engine) -> None:
    """Accept user questions until the user exits."""
    print("\n" + "=" * 50)
    print("Metadata-Enhanced RAG Ready! Type 'exit' or 'quit' to quit.")
    print("=" * 50)

    while True:
        try:
            user_query = input("\nAsk a question about your paper: ").strip()

            if not user_query:
                continue

            if user_query.lower() in EXIT_COMMANDS:
                print("Exiting RAG assistant. Goodbye!")
                break

            print("\n[RAG] Retrieving context with metadata...")
            response = query_engine.query(user_query)
            display_response(response)

        except KeyboardInterrupt:
            print("\nExiting RAG assistant.")
            break

        except Exception as error:
            print(f"\n[Error processing query]: {error}")


def main() -> None: 
    """Initialize and run the Sprint 3 RAG assistant."""
    print("=== Research Paper RAG Assistant — Sprint 3 ===")

    initialize_settings()

    try:
        query_engine = create_query_engine()
    except Exception as error :
        print(f"[Error] Failed to initialize RAG pipeline: {error}")
        return
    if query_engine is None:
        return

    run_query_loop(query_engine)


if __name__ == "__main__": 
    main()

