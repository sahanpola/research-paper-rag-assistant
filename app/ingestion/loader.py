import re
from pathlib import Path
from typing import List

import pymupdf4llm
from llama_index.core import Document, VectorStoreIndex
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


def enrich_document_metadata(pdf_file: Path) -> dict:
    """Return document metadata dictionary enriched with file name and paper title."""
    return {
        "file_name": pdf_file.name,
        "paper_title": pdf_file.stem.replace("_", " ").title(),
    }


def load_documents_from_directory(directory_path: str) -> List[Document]:
    """Load documents from directory using pymupdf4llm for Markdown extraction."""
    path = Path(directory_path)
    if not path.exists():
        raise FileNotFoundError(f"Directory non-existent: {directory_path}")

    pdf_files = list(path.glob("*.pdf"))
    if not pdf_files:
        print(f"[Ingestion] No PDF files found in {directory_path}")
        return []

    print(f"[Ingestion] Loading and parsing {len(pdf_files)} PDFs using pymupdf4llm...")
    
    all_documents = []

    for pdf_file in pdf_files:
        print(f"  -> Parsing (Markdown): {pdf_file.name}")
        
        # Extract entire PDF text as Markdown string
        md_text = pymupdf4llm.to_markdown(str(pdf_file))
        
        # Clean the extracted markdown text
        cleaned_text = clean_pdf_text(md_text)
        
        # Ensure metadata is strictly a standard Python dictionary
        metadata = enrich_document_metadata(pdf_file)
        
        # Construct the LlamaIndex Document correctly
        doc = Document(text=cleaned_text, metadata=metadata)
        all_documents.append(doc)

    print(f"[Ingestion] Successfully loaded and cleaned {len(all_documents)} documents.")
    return all_documents


def build_index_with_metadata(
    documents: List[Document],
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
            "Sample cleaned markdown text preview:\n"
            f"{documents[0].get_content()[:300]}..."
        )



