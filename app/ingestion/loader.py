
import re
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from app.config.settings import initialize_settings


def clean_pdf_text(text: str) -> str:
    """Remove common PDF extraction artifacts and excess blank lines."""
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if re.fullmatch(
            r"[\d.\-\s]+(?:cm|m|l|S|Q|BT|ET|Tf|TJ|Tj)", stripped
        ):
            continue

        if re.fullmatch(r"[QsmSBTl]", stripped):
            continue

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n[ \t]*\n+", "\n\n", cleaned_text)

    return cleaned_text.strip()


def enrich_document_metadata(document: Document) -> dict:
    """Return document metadata enriched with a readable paper title."""
    metadata = document.metadata.copy()
    file_name = metadata.get("file_name")

    if file_name:
        metadata["paper_title"] = (
            Path(file_name).stem.replace("_", " ").title()
        )

    return metadata


def load_documents_from_directory(directory_path: str) -> list[Document]:
    """Load PDFs, clean their text, and enrich their metadata."""
    path = Path(directory_path)

    if not path.is_dir():
        raise NotADirectoryError(
            f"Directory does not exist or is not a directory: {directory_path}"
        )

    print(f"[Ingestion] Loading and cleaning PDFs from: {path}...")

    reader = SimpleDirectoryReader(
        input_dir=str(path),
        required_exts=[".pdf"],
    )
    raw_documents = reader.load_data()

    cleaned_documents = []

    for document in raw_documents:
        cleaned_text = clean_pdf_text(document.get_content())
        metadata = enrich_document_metadata(document)

        cleaned_document = Document(
            text=cleaned_text,
            metadata=metadata,
        )
        cleaned_documents.append(cleaned_document)

    print(
        f"[Ingestion] Loaded and cleaned "
        f"{len(cleaned_documents)} document pages/sources."
    )

    return cleaned_documents


def build_index_with_metadata(
    documents: list[Document],
) -> VectorStoreIndex:
    """Build a vector index from documents and their metadata."""
    print("[Index] Building VectorStoreIndex...")

    index = VectorStoreIndex.from_documents(documents)

    print("[Index] Index built successfully.")
    return index


if __name__ == "__main__":
    initialize_settings()

    documents = load_documents_from_directory("./data/raw")

    if documents:
        print(f"Sample metadata: {documents[0].metadata}")
        print(
            "Sample cleaned text preview:\n"
            f"{documents[0].get_content()[:300]}..."
        )
