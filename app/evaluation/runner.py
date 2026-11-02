import time
from typing import List, Dict, Any
from app.config.settings import initialize_settings
from app.ingestion.loader import load_documents_from_directory
from app.ingestion.chunker import create_chunks
from llama_index.core import VectorStoreIndex
from app.evaluation.dataset import get_evaluation_dataset, EvaluationTestCase
from app.evaluation.metrics import (
    calculate_hit_rate,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg_at_k
)

# Import retrieval & post-processing strategies
from app.retrieval.hybrid_retriever import create_hybrid_retriever
from app.retrieval.query_transformer import create_multi_query_retriever
from app.retrieval.advanced_retriever import create_parent_child_index
from llama_index.core.retrievers import AutoMergingRetriever
from app.retrieval.reranker import get_reranked_nodes


def run_evaluation_suite() -> Dict[str, Any]:
    print("=== Research Paper RAG — Sprint 9: Comprehensive Evaluation Suite ===")
    initialize_settings()

    # 1. Load documents & build common index structures
    docs = load_documents_from_directory("./data/raw")
    if not docs:
        print("[Error] No documents found in data/raw.")
        return {}

    nodes = create_chunks(docs, chunk_size=512, chunk_overlap=50)
    base_index = VectorStoreIndex(nodes)
    
    # Dataset
    dataset = get_evaluation_dataset()
    k = 3

    # Define all 5 strategies to benchmark including Reranked
    strategies = ["Dense", "Hybrid", "Multi-Query", "Parent-Child", "Reranked"]
    
    results_summary = {}

    for strategy in strategies:
        print(f"\n[Benchmarking Strategy: {strategy}]")
        strategy_metrics = []
        
        # Configure retriever based on strategy
        if strategy == "Dense":
            retriever = base_index.as_retriever(similarity_top_k=k)
        elif strategy == "Hybrid":
            retriever = create_hybrid_retriever(base_index, nodes, similarity_top_k=k)
        elif strategy == "Multi-Query":
            retriever = create_multi_query_retriever(base_index, similarity_top_k=k, num_queries=3)
        elif strategy == "Parent-Child":
            pc_index = create_parent_child_index(docs)
            base_pc_retriever = pc_index.as_retriever(similarity_top_k=k)
            retriever = AutoMergingRetriever(
                vector_retriever=base_pc_retriever,
                storage_context=pc_index.storage_context
            )
        elif strategy == "Reranked":
            # For Reranked, we fetch top 20 candidate pool via hybrid retriever and cross-encoder rerank down to top 3
            hybrid_ret = create_hybrid_retriever(base_index, nodes, similarity_top_k=20)
        else:
            continue

        for tc in dataset:
            start_time = time.time()
            
            if strategy == "Reranked":
                # Apply cross-encoder reranker
                reranked_nodes = get_reranked_nodes(
                    hybrid_ret, tc.query, initial_top_k=15, final_top_k=k, model_name="BAAI/bge-reranker-base"
                )
                nodes_with_scores = reranked_nodes
            else:
                nodes_with_scores = retriever.retrieve(tc.query)
                
            retrieval_latency = time.time() - start_time

            # Extract retrieved page labels / metadata
            retrieved_pages = []
            for n in nodes_with_scores:
                # Handle both NodeWithScore objects and raw nodes
                node_obj = n.node if hasattr(n, "node") else n
                meta = node_obj.metadata if hasattr(node_obj, "metadata") else {}
                page = str(meta.get("page_label", meta.get("page", "1")))
                retrieved_pages.append(page)

            # Measure generation latency
            gen_start = time.time()
            query_engine = base_index.as_query_engine(similarity_top_k=k)
            response = query_engine.query(tc.query)
            generation_latency = time.time() - gen_start

            answer_text = str(response)

            # Compute IR Metrics
            hit_rate = calculate_hit_rate(retrieved_pages, tc.relevant_pages)
            prec = calculate_precision_at_k(retrieved_pages, tc.relevant_pages, k)
            rec = calculate_recall_at_k(retrieved_pages, tc.relevant_pages, k)
            mrr = calculate_mrr(retrieved_pages, tc.relevant_pages)
            ndcg = calculate_ndcg_at_k(retrieved_pages, tc.relevant_pages, k)

            # Answer correctness (keyword coverage)
            keywords_matched = sum(1 for kw in tc.expected_keywords if kw.lower() in answer_text.lower())
            answer_correct = keywords_matched >= max(1, len(tc.expected_keywords) // 2)

            strategy_metrics.append({
                "test_id": tc.test_id,
                "query": tc.query,
                "hit_rate": hit_rate,
                "precision": prec,
                "recall": rec,
                "mrr": mrr,
                "ndcg": ndcg,
                "retrieval_latency": retrieval_latency,
                "generation_latency": generation_latency,
                "answer_correct": answer_correct,
                "generated_answer": answer_text,
                "retrieved_chunks_count": len(nodes_with_scores)
            })

        results_summary[strategy] = strategy_metrics

    print("\n[Evaluation] Benchmarking completed across all strategies.")
    return results_summary


if __name__ == "__main__":
    results = run_evaluation_suite()
    print(f"\nEvaluated {len(results)} strategies successfully.")