import os 
from dotenv import load_dotenv
from llama_index.core import Settings 
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama 

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "qwen2.5:3b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
REQUEST_TIMEOUT = 120.0 

def initialize_settings() -> None:
    """Configure LlamaIndex using Ollama models."""
    load_dotenv()

    ollama_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
    llm_model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    Settings.llm = Ollama(
        model=llm_model,
        base_url=ollama_url,
        request_timeout=REQUEST_TIMEOUT,
    )

    Settings.embed_model = OllamaEmbedding(
        model_name=embedding_model,
        base_url=ollama_url,

        )

if __name__ == "__main__":
    initialize_settings() 