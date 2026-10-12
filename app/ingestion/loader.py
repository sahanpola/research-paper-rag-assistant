from pathlib import Path 
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex 
from app.config.settings import initialize_settings 

PDF_EXTENSION = ".pdf"
DEFAULT_DATA_DIRECTORY = "./data/raw"

def load_documents_from_directory(
        directory_path: str,
) -> list[Document]: 
    """Load PDF documents from the specified directory."""
    directory = Path(directory_path)

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Document directory not found:{directory}"
        )
    print(f"[Ingestion] Loading PDFs from:{directory}")

    reader = SimpleDirectoryReader(
        input_dir=str(directory),
        required_exts=[PDF_EXTENSION],
    )

    documents = reader.load_data()

    print(f"[Ingestion] Loaded {len(documents)} document pages/sources.")

    return documents 

def build_basic_index(
        documents: list[Document],
    ) -> VectorStoreIndex: 
    """"Build an in-memory vector index from the provided documents."""
    print("[Index] Generating embeddings and building vector index...")

    index = VectorStoreIndex.from_documents(documents)
    print("[Index] Vector index constructed successfully.")

    return index

def main() -> None:
    """Run the basic document ingestion and indexing workflow."""
    initialize_settings()

    documents = load_documents_from_directory(
        DEFAULT_DATA_DIRECTORY
        )

    if not documents: 
        print("[Ingestion] No documents found. Nothing to index.")
        return

    build_basic_index(documents)

if __name__ == "__main__":
    main()
    






    