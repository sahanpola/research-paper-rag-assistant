from contextlib import asynccontextmanager
from typing import Optional 

from fastapi import FastAPI, HTTPException, status 
from pydantic import BaseModel, Field 
from llama_index.core import VectorStoreIndex

from app.config.settings import initialize_settings
from app.ingestion.loader import load_documents_from_directory
from app.ingestion.chunker import create_chunks
from app.generation.citation_builder import format_response_with_citations

#1. Application Resources 

index: Optional[VectorStoreIndex] = None

#2.Initalize the RAG pipeline 

def initialize_rag_pipeline() -> Optional[VectorStoreIndex]:
     """Load documents, create chunks, and build the vector index."""

     print("[API] Initializing RAG pipeline...")

     initialize_settings()
     documents = load_documents_from_directory("./data/raw")

     if not documents:
            print("[API] No documents found in './data/raw'. Add a PDF first.")
            return None

     nodes = create_chunks(documents, 
                           chunk_size=512,
                            chunk_overlap=50)

     vector_index = VectorStoreIndex(nodes)
     print("[API] RAG pipeline initialized successfully.")
     return vector_index 

#3.Application Lifecspan 

@asynccontextmanager
async def lifespan(app: FastAPI):
      """ Initialize resources when the API starts."""
      global index

      index = initialize_rag_pipeline()

      yield 
      # Cleanup resources if required in the future.
      index = None


#4.Create FastAPI application 

app = FastAPI(
    title="Research Paper RAG API",
    version="1.0.0",
    description=(
        "A Research Paper RAG API powered by "
        "LlamaIndex and local Ollama models."
    ),
    lifespan=lifespan,
)

#5.Request and response schema 

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Question to ask the research paper assistant.",
    )

    similarity_top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve.",
    )


class CitationResponse(BaseModel):
    query: str
    response_with_citations: str

# 6. Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check() -> dict:
    """Check API and RAG index status."""

    return {
        "status": "healthy",
        "index_loaded": index is not None,
    }


# --------------------------------------------------
# 7. RAG query endpoint
# --------------------------------------------------

@app.post(
    "/query",
    response_model=CitationResponse,
    status_code=status.HTTP_200_OK,
)
def query_rag(payload: QueryRequest) -> CitationResponse:
    """Retrieve relevant research paper chunks and generate an answer."""

    if index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG index is not available.",
        )

    try:
        query_engine = index.as_query_engine(
            similarity_top_k=payload.similarity_top_k,
        )

        response = query_engine.query(payload.query)

        formatted_response = format_response_with_citations(response)

        return CitationResponse(
            query=payload.query,
            response_with_citations=formatted_response,
        )

    except Exception:
        # Log the full exception on the server.
        import logging

        logging.exception("[API Error] RAG query failed.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your query.",
        )

      
