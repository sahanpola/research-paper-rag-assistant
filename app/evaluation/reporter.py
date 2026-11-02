from typing import Dict, Any


def generate_evaluation_report(results_summary: Dict[str, Any]) -> None:
    """
    Generate a clean, readable terminal report summarizing strategy benchmarks,
    IR metrics, latencies, and generated answers.
    """
    print("\n" + "=" * 70)
    print("               RESEARCH PAPER RAG — EVALUATION REPORT")
    print("=" * 70)

    for strategy, metrics_list in results_summary.items():
        print(f"\n--- Strategy: {strategy} ---")
        
        if not metrics_list:
            print("  No metrics available.")
            continue

        total_tests = len(metrics_list)
        avg_hit_rate = sum(m["hit_rate"] for m in metrics_list) / total_tests
        avg_prec = sum(m["precision"] for m in metrics_list) / total_tests
        avg_rec = sum(m["recall"] for m in metrics_list) / total_tests
        avg_mrr = sum(m["mrr"] for m in metrics_list) / total_tests
        avg_ndcg = sum(m["ndcg"] for m in metrics_list) / total_tests
        avg_ret_lat = sum(m["retrieval_latency"] for m in metrics_list) / total_tests
        avg_gen_lat = sum(m["generation_latency"] for m in metrics_list) / total_tests
        passed_answers = sum(1 for m in metrics_list if m["answer_correct"])

        print(f"  Test Cases Evaluated  : {total_tests}")
        print(f"  Hit Rate @ 3          : {avg_hit_rate:.2f}")
        print(f"  Precision @ 3         : {avg_prec:.2f}")
        print(f"  Recall @ 3            : {avg_rec:.2f}")
        print(f"  MRR                   : {avg_mrr:.2f}")
        print(f"  NDCG @ 3              : {avg_ndcg:.2f}")
        print(f"  Answer Correctness    : {passed_answers}/{total_tests} passed")
        print(f"  Avg Retrieval Latency : {avg_ret_lat:.4f}s")
        print(f"  Avg Generation Latency: {avg_gen_lat:.4f}s")
        
        print("\n  Detailed Test Breakdown:")
        for m in metrics_list:
            status = "PASS" if m["answer_correct"] else "FAIL"
            print(f"    [{m['test_id']}] Status: {status} | Hit: {m['hit_rate']} | MRR: {m['mrr']:.2f}")
            print(f"         Query: {m['query']}")
            print(f"         Answer: {m['generated_answer'].strip()[:120]}...\n")

    print("=" * 70)
    print("Evaluation report generated successfully.")
    print("=" * 70)


if __name__ == "__main__":
    from app.evaluation.runner import run_evaluation_suite
    results = run_evaluation_suite()
    if results:
        generate_evaluation_report(results)