from ast import While
import sys 
from llama_index.core import VectorStoreIndex
from app.config.settings import initialize_settings
from app.ingestion.loader import (
    build_basic_index,
    load_documents_from_directory,
)

DATA_DIRECTORY = "./data/raw"
TOP_K_RESULTS = 3
EXIT_COMMANDS={"exit","quit"}


def initialize_application() -> None:
    """Initialize the application settings."""
    try:
        initialize_settings()
    except Exception as error:
        print(f"[Configuration Error] Failed to initialize settings: {error}")
        sys.exit(1)

def load_documents():
    """Load PDF documents from the configured data directory."""
    try:
        documents = load_documents_from_directory(DATA_DIRECTORY)
    except Exception as error:
        print(f"[Error] Failed to load documents: {error}")
        sys.exit(1)

    if not documents:
        print(
             f"[Warning] No PDF files found in '{DATA_DIRECTORY}'. " 
              "Please add a sample PDF to run Sprint 1." ) 
        sys.exit(0)
    return documents

def create_query_engine(documents) :
    """Build the vector index and create a query engine."""
    index: VectorStoreIndex = build_basic_index(documents)

    return index.as_query_engine(
        similarity_top_k=TOP_K_RESULTS,
    
    )

def display_welcome_message() -> None:
    """Display the RAG assistant welcome meddage"""
    print("\n" + "=" * 50)
    print("RAG System Ready! Type 'exit' or 'quit' to stop.") 
    print("=" * 50)

def is_exit_command(user_query: str) -> bool:
    """Check whether the user wants to exit the application."""
    return user_query.lower() in EXIT_COMMANDS

def display_response(response) -> None:
    """Display the generated answer and its retrieved sources."""
    print("\n" + "-" * 20 + " ANSWER " + "-" * 20)
    print(str(response).strip())
    print("-" * 48)

    display_sources(response)

def display_sources(response) -> None:
    """Display the source nodes used to generate the response."""
    print("\n[Retrieved Context Sources]:")

    for index, node in enumerate(response.source_nodes,start=1):
        score = (
            f"{node.score:4f}"
            if node.score is not None
            else "N/A"
        )

        file_name = node.metadata.get("file_name", "Unknown")
        page_label = node.metadata.get("page_label", "NA")


        print(
            f" ({index}) "
            f"File: {file_name} | "
            f"Page: {page_label} | "
            f"Score: {score}"

        )

def run_query_loop(query_engine) -> None:
    """Run the interactive question and answer loop """ 
    while True:

        try:
            user_query = input(
                "\nAsk a question about your paper: " 
            ).strip()

            if not user_query:
                continue

            if is_exit_command(user_query):
                 print("Exiting RAG assistant. Goodbye!")
                 break

            print( 
               "\n[RAG] Retrieving context & synthesizing answer..." 
               ) 

            response = query_engine.query(user_query)
            display_response(response)

        except KeyboardInterrupt:
            print("\nExiting...")
            break

        except Exception as error:
            print(f"[Error] An error occurred: {error}")


def main() -> None: 
    """"Run the Sprint 1 basic application """
    print(
        "=== Research Paper RAG Assistant — "
        "Sprint 1 (Basic RAG) ==="

    )

initialize_application()
documents = load_documents()
query_engine = create_query_engine(documents)
display_welcome_message()
run_query_loop(query_engine)




print("\n=== Sprint 0 Setup Verified Successfully! ===")

if __name__ == "__main__": 
    main()

