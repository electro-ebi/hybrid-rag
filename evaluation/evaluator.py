"""
Basic evaluator for RAG system.
"""

from typing import List, Dict, Any
import time
from logger import get_logger

logger = get_logger("evaluator")

class Evaluator:
    """Basic evaluator for RAG system performance."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_retrieval(self, query: str, results: List[Dict], expected: List[str] = None) -> Dict[str, Any]:
        """Evaluate retrieval performance."""
        start_time = time.time()
        
        evaluation = {
            'query': query,
            'results_count': len(results),
            'latency_ms': (time.time() - start_time) * 1000,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if expected:
            # Basic precision/recall calculation
            retrieved_docs = [r.get('source', '') for r in results]
            relevant_retrieved = len(set(retrieved_docs) & set(expected))
            
            evaluation.update({
                'precision': relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0,
                'recall': relevant_retrieved / len(expected) if expected else 0,
                'expected_count': len(expected),
                'relevant_retrieved': relevant_retrieved
            })
        
        logger.info(f"Evaluation completed for query: {query}")
        return evaluation
    
    def evaluate_generation(self, query: str, response: str, context: List[str] = None) -> Dict[str, Any]:
        """Evaluate generation performance."""
        evaluation = {
            'query': query,
            'response_length': len(response),
            'context_count': len(context) if context else 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Basic metrics
        evaluation.update({
            'has_response': bool(response.strip()),
            'response_words': len(response.split()),
            'avg_response_length': len(response) / max(len(response.split()), 1)
        })
        
        return evaluation
    
    def run_batch_evaluation(self, queries: List[str], retriever_func, expected_results: List[List[str]] = None) -> Dict[str, Any]:
        """Run batch evaluation."""
        batch_results = []
        
        for i, query in enumerate(queries):
            try:
                # Test retrieval
                results = retriever_func(query, top_k=5)
                expected = expected_results[i] if expected_results and i < len(expected_results) else None
                
                eval_result = self.evaluate_retrieval(query, results, expected)
                batch_results.append(eval_result)
                
            except Exception as e:
                logger.error(f"Error evaluating query {i}: {e}")
                batch_results.append({
                    'query': query,
                    'error': str(e),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Aggregate metrics
        successful_evals = [r for r in batch_results if 'error' not in r]
        
        aggregate = {
            'total_queries': len(queries),
            'successful_evaluations': len(successful_evals),
            'failed_evaluations': len(batch_results) - len(successful_evals),
            'avg_latency_ms': sum(r.get('latency_ms', 0) for r in successful_evals) / len(successful_evals) if successful_evals else 0,
            'avg_results_count': sum(r.get('results_count', 0) for r in successful_evals) / len(successful_evals) if successful_evals else 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add precision/recall averages if available
        if any('precision' in r for r in successful_evals):
            aggregate['avg_precision'] = sum(r.get('precision', 0) for r in successful_evals) / len(successful_evals)
            aggregate['avg_recall'] = sum(r.get('recall', 0) for r in successful_evals) / len(successful_evals)
        
        return {
            'aggregate_metrics': aggregate,
            'individual_results': batch_results
        }
