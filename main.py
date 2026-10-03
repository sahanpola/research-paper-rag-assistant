import sys 
from llama_index.core import Settings
from app.config.settings import initialize_settings

def initialize_application() -> None:
    """Initialize the application settings."""
    try:
        initialize_settings()
    except Exception as error:
        print(f"[Configuration Error] Failed to initialize settings: {error}")
        sys.exit(1)

def test_llm_connection() -> None:
    """Verify that the configured LLM is reachable.""" 
    print("\n[Test] Testing LLM connection...")

    try: 
        response = Settings.llm.complete(
            "Say 'Sprint 0 Ready' if you received this."
        )
        print(f"[LLM Response] {response.text.strip()}")

    except Exception as error:
        print(
            "[LLM Error] Could not reach Ollama."
            f"Is Ollama running? Error: {error}"
        )
        sys.exit(1)

def test_embedding_connection() -> None:
    """Verify that the configured embedding model is reachable.""" 
    print("\n[Test] Testing embedding model connection...")

    try:
        embedding = Settings.embed_model.get_text_embedding(
            "Research Paper RAG Test"
        )
        print(
            "[Embedding Success]"
            f"Vector generated with dimension length {len(embedding)}"
        )
    except Exception as error:
        print( "[Embedding Error] "
               f"Failed to generate embedding. Error: {error}"
                 )
        sys.exit(1)

def main() -> None: 
    """"Run the Sprint 0 application sanity checks."""
print("=== Research Paper RAG Assistant - Sprint 0 Sanity Check ===")

initialize_application()
test_llm_connection()
test_embedding_connection()

print("\n=== Sprint 0 Setup Verified Successfully! ===")

if __name__ == "__main__": 
    main()

