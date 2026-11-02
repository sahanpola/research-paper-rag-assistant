import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from app.config.settings import initialize_settings
from app.ingestion.loader import load_documents_from_directory
from app.ingestion.chunker import create_chunks
from llama_index.core import VectorStoreIndex
from app.generation.citation_builder import format_response_with_citations

app = FastAPI(
    title="Research Paper RAG Assistant API",
    version="1.1.0",
    description="Production-grade RAG API backed by local Ollama and LlamaIndex with file management."
)

UPLOAD_DIR = "./data/raw"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global index storage
index = None

def reload_rag_index():
    global index
    print("[API] Reloading RAG index from directory...")
    initialize_settings()
    docs = load_documents_from_directory(UPLOAD_DIR)
    if docs:
        nodes = create_chunks(docs, chunk_size=512, chunk_overlap=50)
        index = VectorStoreIndex(nodes)
        print(f"[API] RAG Index reloaded successfully with {len(docs)} documents.")
    else:
        index = None
        print("[API Warning] No documents found in ./data/raw.")

@app.on_event("startup")
def startup_event():
    reload_rag_index()


class QueryRequest(BaseModel):
    query: str
    similarity_top_k: Optional[int] = 3


class CitationResponse(BaseModel):
    query: str
    response_with_citations: str


@app.get("/health")
def health_check():
    return {"status": "healthy", "index_loaded": index is not None}


@app.get("/files", response_model=List[str])
def list_files():
    if not os.path.exists(UPLOAD_DIR):
        return []
    return [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Rebuild RAG index with new file
    reload_rag_index()
    return {"filename": file.filename, "message": "File uploaded and RAG index reloaded successfully."}


@app.delete("/files/{filename}")
def delete_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    
    os.remove(file_path)
    # Rebuild RAG index after deletion
    reload_rag_index()
    return {"filename": filename, "message": "File deleted and RAG index reloaded successfully."}


@app.post("/query", response_model=CitationResponse)
def query_rag(payload: QueryRequest):
    if index is None:
        raise HTTPException(status_code=500, detail="RAG index is not initialized or no documents are available.")
    
    try:
        query_engine = index.as_query_engine(similarity_top_k=payload.similarity_top_k)
        response = query_engine.query(payload.query)
        formatted_response = format_response_with_citations(response)
        
        return CitationResponse(
            query=payload.query,
            response_with_citations=formatted_response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))