from dataclasses import dataclass, field
from typing import List


@dataclass
class EvaluationTestCase:
    test_id: str
    query: str
    reference_answer: str
    expected_keywords: List[str] = field(default_factory=list)
    relevant_pages: List[str] = field(default_factory=list)


def get_evaluation_dataset() -> List[EvaluationTestCase]:
    """
    Returns a structured golden evaluation dataset based on our loaded research papers:
    1. Process Mining (Data Science in Action)
    2. LLM-Based Exogenous World Modeling for Dynamic Scheduling (AWMP)
    """
    return [
        EvaluationTestCase(
            test_id="TC-01",
            query="What is data science and what disciplines does it amalgamate?",
            reference_answer="Data science is a new and important discipline that can be viewed as an amalgamation of classical disciplines like statistics, data mining, databases, and distributed systems[cite: 1].",
            expected_keywords=["data science", "statistics", "data mining", "databases", "distributed systems"],
            relevant_pages=["1"]
        ),
        EvaluationTestCase(
            test_id="TC-02",
            query="What is the internet of events and how did society shift?",
            reference_answer="Society shifted from being predominantly analog to digital in just a few years, leading to data being collected about anything at any time, characterized as the internet of events[cite: 1].",
            expected_keywords=["internet of events", "analog", "digital", "always on"],
            relevant_pages=["1", "2"]
        ),
        EvaluationTestCase(
            test_id="TC-03",
            query="What is Afterstate World-Model Planning (AWMP) used for in dynamic scheduling?",
            reference_answer="AWMP is an afterstate world-model planning framework for dynamic scheduling that separates known deterministic shop transitions from uncertain exogenous job arrivals, using an LLM to forecast arrival belief and short scenario rollouts for dispatching[cite: 3, 4].",
            expected_keywords=["AWMP", "afterstate", "world model", "dynamic scheduling", "arrival belief"],
            relevant_pages=["3", "4", "5"]
        )
    ]


if __name__ == "__main__":
    dataset = get_evaluation_dataset()
    print(f"[Dataset] Loaded {len(dataset)} structured evaluation test cases successfully across multiple papers.")
    for tc in dataset:
        print(f"  - [{tc.test_id}] {tc.query}")