from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter 
from llama_index.core.schema import TextNode 

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
SAMPLE_CHUNK_SIZE = 256
SAMPLE_CHUNK_OVERLAP = 30
PREVIEW_LENGTH = 200

def create_chunks(
        documents: list[Document],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        ) -> list[TextNode]:
    """Split documents into overlapping text chunks."""

    if chunk_size <=0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller"
                         " than chunk_size")


    print(
        f"[Chunker] Creating with size={chunk_size},"
        f" overlap={chunk_overlap}..."
    )  

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    nodes = splitter.get_nodes_from_documents(documents)
    print(
        f"[Chunker] Generated {len(nodes)} chunks "
        f"from {len(documents)} documents."
    )

    return nodes

def main() -> None:
    """ Run a basic test of the document chunking process """
    from app.config.settings import initialize_settings
    from app.ingestion.loader import load_documents_from_directory

    initialize_settings()
    documents = load_documents_from_directory("./data/raw")

    if not documents:
        print("[Chunker] No documents found. Nothing to chunk.")
        return

    nodes = create_chunks(
        documents,
        chunk_size=SAMPLE_CHUNK_SIZE,
        chunk_overlap=SAMPLE_CHUNK_OVERLAP,
    )

    if nodes:
        preview = nodes[0].text[:PREVIEW_LENGTH]
        print(f"\nSample chunk preview:\n{preview}...")

if __name__ == "__main__":
    main()



    