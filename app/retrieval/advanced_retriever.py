from typing import List 

from llama_index.core import Document,StorageContext,VectorStoreIndex 
from llama_index.core.node_parser import (HierarchicalNodeParser, get_leaf_nodes,)

from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore 

from app.config.settings import initialize_settings 
from app.ingestion.loader import load_documents_from_directory 

# Configuration 
PARENT_CHUNK_SIZE = 2048
INTERMEDIATE_CHUNK_SIZE = 512
CHILD_CHUNK_SIZE = 128 

DEFAULT_TOP_K = 2
TEST_QUERY = "What is data science?"



def create_parent_child_index(
        documents: List[Document],
) -> VectorStoreIndex:
    """Build a hierarchical index using child nodes for precise retrieval."""

    print("[AdvancedRAG] Creating hierarchical nodes...")

    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[
            PARENT_CHUNK_SIZE,
            INTERMEDIATE_CHUNK_SIZE,
            CHILD_CHUNK_SIZE,
        ]
    )

    all_nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)

    print(
        f"[AdvancedRAG] Created {len(all_nodes)} hierarchical nodes "
        f"and {len(leaf_nodes)} leaf nodes."
    )

    # Store all nodes so parent nodes can be accessed later.
    docstore = SimpleDocumentStore()
    docstore.add_documents(all_nodes)

    storage_context = StorageContext.from_defaults(
        docstore=docstore
    )

    #Index only the smallest child nodes.
    index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
    )
    print("[AdvancedRAG] Parent-child index built successfully.")

    return index


def create_parent_child_retriever(
        index: VectorStoreIndex,
        similarity_top_k: int = DEFAULT_TOP_K,

) -> AutoMergingRetriever:
    """Create a retriever that merges relevant child nodes into parent context."""

    print("[AdvancedRAG] Initializing AutoMergingRetriever...")

    base_retriever = index.as_retriever(
        similarity_top_k=similarity_top_k
    )

    merging_retriever = AutoMergingRetriever(
        vector_retriever=base_retriever,
        storage_context=index.storage_context,
        verbose=True,
    )

    print("[AdvancedRAG] Parent-child retriever initialized.")

    return merging_retriever

def run_parent_child_experiment() -> None: 
    """Load research papers, build a hierarchical index, and test retrieval."""
    initialize_settings()
    documents = load_documents_from_directory("./data/raw")

    if not documents:
        print("[Error] No documents found in ./data/raw.")
        return
    index = create_parent_child_index(documents)

    retriever = create_parent_child_retriever(
        index=index,
        similarity_top_k=DEFAULT_TOP_K,
    )

    results = retriever.retrieve(TEST_QUERY)

    print(f"\nQuery: {TEST_QUERY}")
    print(f"Retrieved {len(results)} results.\n")

    for rank, result in enumerate(results, start=1):
        content = result.node.get_content().strip()
        score = result.score
        score_text = f"{score:.4f}" if score is not None else "N/A"

        print(f"[{rank}] Score: {score_text}")
        print(f"Node ID: {result.node.node_id}")
        print(f"Node type: {result.node.metadata.get('node_type', 'N/A')}")
        print(f"Content: {content[:300]}...")
        print("-" * 60)


if __name__ == "__main__":
    run_parent_child_experiment()










    




