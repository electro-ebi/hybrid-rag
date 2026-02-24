"""
Run evaluation for RAG system.
"""

import json
from typing import List, Dict, Any
from .evaluator import Evaluator
from logger import get_logger

logger = get_logger("run_evaluation")

def run_evaluation(retriever_func, queries: List[str], expected_results: List[List[str]] = None) -> Dict[str, Any]:
    """
    Run evaluation on RAG system.
    
    Args:
        retriever_func: Function that takes (query, top_k) and returns results
        queries: List of test queries
        expected_results: List of expected relevant documents for each query
    
    Returns:
        Evaluation results dictionary
    """
    evaluator = Evaluator()
    
    logger.info(f"Starting evaluation with {len(queries)} queries")
    
    results = evaluator.run_batch_evaluation(queries, retriever_func, expected_results)
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Evaluation completed and saved to evaluation_results.json")
    
    return results

def main():
    """Main evaluation function."""
    # Sample queries for testing
    test_queries = [
        "hybrid retrieval systems",
        "vector embeddings",
        "document chunking",
        "semantic search",
        "RAG evaluation"
    ]
    
    # This would be replaced with actual retriever function
    def sample_retriever(query: str, top_k: int = 5):
        """Sample retriever for testing."""
        return [
            {'source': f'doc_{i}', 'content': f'Sample content for {query}', 'score': 1.0 - i*0.1}
            for i in range(min(top_k, 3))
        ]
    
    results = run_evaluation(sample_retriever, test_queries)
    
    print("Evaluation Results:")
    print(f"Total queries: {results['aggregate_metrics']['total_queries']}")
    print(f"Successful: {results['aggregate_metrics']['successful_evaluations']}")
    print(f"Avg latency: {results['aggregate_metrics']['avg_latency_ms']:.2f}ms")

if __name__ == "__main__":
    main()
