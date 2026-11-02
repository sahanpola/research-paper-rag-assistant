from typing import Any 
from llama_index.core.base.response.schema import RESPONSE_TYPE

from llama_index.core import VectorStoreIndex
from app.config.settings import initialize_settings
from app.ingestion.loader import load_documents_from_directory
from app.ingestion.chunker import create_chunks

def extract_retrieved_chunks(response: Any) -> list[RESPONSE_TYPE]:
    """ 
     Extract unique source citations from a LlamaIndex response.

    Each citation contains:
        - paper_title
        - file_name
        - page
        - relevance_score
    """
    citations = []
    seen_sources = set() 

    for source_node in response.source_nodes:
        node = source_node.node
        metadata = node.metadata

        file_name = (
            metadata.get("file_name")
            or metadata.get("filename")
            or "Unknown File"
        )

        paper_title = (
            metadata.get("paper_title")
            or metadata.get("title")
            or file_name
        )

        page = (
            metadata.get("page_label")
            or metadata.get("page")
            or metadata.get("page_number")
            or "N/A"
        )

        # Avoid duplicate citations for the same paper and page.
        source_key = (file_name, str(page))

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)

        citations.append({
            "paper_title": paper_title,
            "file_name": file_name,
            "page": str(page),
            "relevance_score": source_node.score,
        })

    return citations


def format_citations(citations: list[dict[str,Any]]) -> str:
     """Format extracted citations into readable text."""

     if not citations:
        return "No direct source citations available."

     citation_lines = []

     for index, citation in enumerate(citations, start=1):

        score = citation["relevance_score"]

        score_text = (
            f"{score:.4f}"
            if score is not None
            else "N/A"
        )

        citation_lines.append(
            f"[{index}] "
            f"Paper: {citation['paper_title']} | "
            f"File: {citation['file_name']} | "
            f"Page: {citation['page']} | "
            f"Relevance Score: {score_text}"
        )

     return "\n".join(citation_lines)

def format_response_with_citations(
    response: RESPONSE_TYPE,
) -> str:
    """
    Combine the generated answer with its source citations.
    """

    answer_text = str(response).strip()

    citations = extract_retrieved_chunks(response)
    formatted_citations = format_citations(citations)

    return (
        f"{answer_text}\n\n"
        f"{'=' * 55}\n"
        f"VERIFIED SOURCES & CITATIONS\n"
        f"{'-' * 55}\n"
        f"{formatted_citations}\n"
        f"{'=' * 55}"
    )

if __name__ == "__main__":
    initialize_settings()

    documents = load_documents_from_directory("./data/raw")

    if not documents:
        print("No documents found.")
    else:
        nodes = create_chunks(
            documents,
            chunk_size=512,
            chunk_overlap=50,
        )

        index = VectorStoreIndex(nodes)

        query_engine = index.as_query_engine(
            similarity_top_k=2,
        )

        response = query_engine.query(
            "What is data science?"
        )

        print(format_response_with_citations(response))