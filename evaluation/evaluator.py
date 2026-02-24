"""
Rigorous evaluation framework for autonomous AI agent.
Measures retrieval quality, latency, and ablation studies.
"""
from __future__ import annotations

import json
import time
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import re

from data_layer.retrieval import retrieve_chunks
from data_layer.cross_encoder_reranker import rerank_chunks
from llm import call_llm
from logger import get_logger

logger = get_logger("evaluator")

@dataclass
class EvaluationResult:
    """Single evaluation result."""
    question_id: str
    question: str
    domain: str
    retrieved_chunks: List[str]
    generated_answer: str
    expected_keywords: List[str]
    precision_at_k: float
    recall_at_k: float
    mrr: float
    latency_breakdown: Dict[str, float]
    keyword_matches: List[str]

class Evaluator:
    """Rigorous evaluation framework."""
    
    def __init__(self):
        self.results: List[EvaluationResult] = []
        
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load evaluation dataset."""
        with open(dataset_path, 'r') as f:
            return json.load(f)
    
    def evaluate_retrieval_only(
        self, 
        dataset_path: str,
        top_k: int = 5,
        modes: List[str] = ["dense", "sparse", "hybrid", "hybrid_rerank"]
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval performance ONLY (no LLM generation).
        This isolates retrieval quality from generation bottlenecks.
        
        Args:
            dataset_path: Path to evaluation dataset
            top_k: Number of chunks to retrieve
            modes: Retrieval modes to evaluate
            
        Returns:
            Retrieval-only evaluation results
        """
        logger.info("Starting RETRIEVAL-ONLY evaluation with %d questions", len(modes))
        
        dataset = self.load_dataset(dataset_path)
        mode_results = {mode: [] for mode in modes}
        
        # Pre-load cross-encoder reranker to avoid per-question loading
        logger.info("Pre-loading cross-encoder reranker...")
        from data_layer.cross_encoder_reranker import get_reranker
        reranker = get_reranker()
        logger.info("Cross-encoder loaded and ready")
        
        for question_item in dataset:
            question = question_item["question"]
            expected_keywords = question_item["expected_keywords"]
            domain = question_item["domain"]
            
            logger.info("Evaluating retrieval for: %s", question[:50])
            
            for mode in modes:
                result = self._evaluate_retrieval_single_question(
                    question, expected_keywords, domain, mode, top_k, reranker
                )
                mode_results[mode].append(result)
        
        # Aggregate results by mode
        final_results = {}
        for mode, results in mode_results.items():
            final_results[mode] = self._aggregate_retrieval_results(results, mode)
        
        # Generate comparison tables
        comparison = self._generate_retrieval_comparison_table(final_results)
        
        return {
            "mode_results": final_results,
            "comparison_table": comparison,
            "dataset_size": len(dataset),
            "evaluation_type": "retrieval_only",
            "evaluation_config": {
                "top_k": top_k,
                "modes": modes
            }
        }
    
    def _evaluate_retrieval_single_question(
        self,
        question: str,
        expected_keywords: List[str],
        domain: str,
        mode: str,
        top_k: int,
        reranker
    ) -> EvaluationResult:
        """Evaluate single question retrieval ONLY (no LLM generation)."""
        
        # Start timing
        start_time = time.perf_counter()
        latency_breakdown = {}
        
        try:
            # Retrieval phase
            retrieval_start = time.perf_counter()
            
            if mode == "dense":
                chunks = retrieve_chunks(question, top_k=top_k, mode="dense")
            elif mode == "sparse":
                chunks = retrieve_chunks(question, top_k=top_k, mode="sparse")
            elif mode == "hybrid":
                chunks = retrieve_chunks(question, top_k=top_k, mode="hybrid")
            elif mode == "hybrid_rerank":
                # First get hybrid results, then rerank
                chunks = retrieve_chunks(question, top_k=top_k*2, mode="hybrid")
                chunks = reranker.rerank(question, chunks)
                chunks = chunks[:top_k]
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            retrieval_time = time.perf_counter() - retrieval_start
            latency_breakdown["retrieval"] = retrieval_time
            
            # NO GENERATION - Skip LLM call entirely
            generated_answer = ""  # Empty for retrieval-only evaluation
            generation_time = 0.0
            latency_breakdown["generation"] = generation_time
            
            # Calculate metrics based on retrieved chunks only
            precision_at_k, recall_at_k, mrr, keyword_matches = self._calculate_retrieval_metrics(
                chunks, expected_keywords
            )
            
            total_time = time.perf_counter() - start_time
            latency_breakdown["total"] = total_time
            
            return EvaluationResult(
                question_id=f"{mode}_retrieval_{int(time.time())}",
                question=question,
                domain=domain,
                retrieved_chunks=chunks,
                generated_answer=generated_answer,
                expected_keywords=expected_keywords,
                precision_at_k=precision_at_k,
                recall_at_k=recall_at_k,
                mrr=mrr,
                latency_breakdown=latency_breakdown,
                keyword_matches=keyword_matches
            )
            
        except Exception as e:
            logger.error("Retrieval evaluation failed for question: %s, error: %s", question[:50], e)
            
            # Return failed result
            total_time = time.perf_counter() - start_time
            return EvaluationResult(
                question_id=f"{mode}_failed_{int(time.time())}",
                question=question,
                domain=domain,
                retrieved_chunks=[],
                generated_answer="",
                expected_keywords=expected_keywords,
                precision_at_k=0.0,
                recall_at_k=0.0,
                mrr=0.0,
                latency_breakdown={"total": total_time, "error": str(e)},
                keyword_matches=[]
            )
    
    def _calculate_retrieval_metrics(
        self,
        retrieved_chunks: List[str],
        expected_keywords: List[str]
    ) -> Tuple[float, float, float, List[str]]:
        """Calculate retrieval metrics based on chunks only (no generation)."""
        
        # Check keyword matches in retrieved chunks
        keyword_matches = []
        found_keywords = set()
        
        for chunk in retrieved_chunks:
            chunk_lower = chunk.lower()
            for keyword in expected_keywords:
                if keyword.lower() in chunk_lower:
                    keyword_matches.append(keyword)
                    found_keywords.add(keyword.lower())
        
        # Precision@k: fraction of relevant chunks among retrieved
        relevant_chunks = 0
        for chunk in retrieved_chunks:
            chunk_lower = chunk.lower()
            if any(keyword.lower() in chunk_lower for keyword in expected_keywords):
                relevant_chunks += 1
        
        precision_at_k = relevant_chunks / len(retrieved_chunks) if retrieved_chunks else 0.0
        
        # Recall@k: fraction of expected keywords found in retrieved chunks
        recall_at_k = len(found_keywords) / len(expected_keywords) if expected_keywords else 0.0
        
        # MRR (Mean Reciprocal Rank): 1/rank of first relevant chunk
        mrr = 0.0
        for i, chunk in enumerate(retrieved_chunks):
            chunk_lower = chunk.lower()
            if any(keyword.lower() in chunk_lower for keyword in expected_keywords):
                mrr = 1.0 / (i + 1)
                break
        
        return precision_at_k, recall_at_k, mrr, list(found_keywords)
    
    def _aggregate_retrieval_results(self, results: List[EvaluationResult], mode: str) -> Dict[str, Any]:
        """Aggregate retrieval-only results across all questions for a mode."""
        
        if not results:
            return {"error": "No results to aggregate"}
        
        # Calculate averages
        precision_scores = [r.precision_at_k for r in results]
        recall_scores = [r.recall_at_k for r in results]
        mrr_scores = [r.mrr for r in results]
        
        # Latency aggregation (retrieval only)
        retrieval_times = []
        total_times = []
        
        for r in results:
            if "retrieval" in r.latency_breakdown:
                retrieval_times.append(r.latency_breakdown["retrieval"])
            if "total" in r.latency_breakdown:
                total_times.append(r.latency_breakdown["total"])
        
        # Domain-specific performance
        domain_performance = {}
        for result in results:
            domain = result.domain
            if domain not in domain_performance:
                domain_performance[domain] = []
            domain_performance[domain].append(result.precision_at_k)
        
        # Calculate domain averages
        domain_averages = {
            domain: sum(scores) / len(scores) 
            for domain, scores in domain_performance.items()
        }
        
        return {
            "mode": mode,
            "total_questions": len(results),
            "precision_at_k": {
                "mean": sum(precision_scores) / len(precision_scores),
                "scores": precision_scores
            },
            "recall_at_k": {
                "mean": sum(recall_scores) / len(recall_scores),
                "scores": recall_scores
            },
            "mrr": {
                "mean": sum(mrr_scores) / len(mrr_scores),
                "scores": mrr_scores
            },
            "latency": {
                "retrieval_ms": {
                    "mean": sum(retrieval_times) / len(retrieval_times) * 1000 if retrieval_times else 0,
                    "scores": [t * 1000 for t in retrieval_times]
                },
                "total_ms": {
                    "mean": sum(total_times) / len(total_times) * 1000 if total_times else 0,
                    "scores": [t * 1000 for t in total_times]
                }
            },
            "domain_performance": domain_averages,
            "keyword_match_rate": sum(len(r.keyword_matches) / len(r.expected_keywords) for r in results if r.expected_keywords) / len(results)
        }
    
    def _generate_retrieval_comparison_table(self, mode_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate retrieval-only comparison table across modes."""
        
        table = {
            "precision_at_k": {},
            "recall_at_k": {},
            "mrr": {},
            "latency_retrieval_ms": {},
            "keyword_match_rate": {}
        }
        
        for mode, results in mode_results.items():
            if "error" in results:
                continue
                
            table["precision_at_k"][mode] = results["precision_at_k"]["mean"]
            table["recall_at_k"][mode] = results["recall_at_k"]["mean"]
            table["mrr"][mode] = results["mrr"]["mean"]
            table["latency_retrieval_ms"][mode] = results["latency"]["retrieval_ms"]["mean"]
            table["keyword_match_rate"][mode] = results["keyword_match_rate"]
        
        return table
    
    def _aggregate_results(self, results: List[EvaluationResult], mode: str) -> Dict[str, Any]:
        """Aggregate results across all questions for a mode."""
        
        if not results:
            return {"error": "No results to aggregate"}
        
        # Calculate averages
        precision_scores = [r.precision_at_k for r in results]
        recall_scores = [r.recall_at_k for r in results]
        mrr_scores = [r.mrr for r in results]
        
        # Latency aggregation
        retrieval_times = []
        generation_times = []
        total_times = []
        
        for r in results:
            if "retrieval" in r.latency_breakdown:
                retrieval_times.append(r.latency_breakdown["retrieval"])
            if "generation" in r.latency_breakdown:
                generation_times.append(r.latency_breakdown["generation"])
            if "total" in r.latency_breakdown:
                total_times.append(r.latency_breakdown["total"])
        
        # Domain-specific performance
        domain_performance = {}
        for result in results:
            domain = result.domain
            if domain not in domain_performance:
                domain_performance[domain] = []
            domain_performance[domain].append(result.precision_at_k)
        
        # Calculate domain averages
        domain_averages = {
            domain: sum(scores) / len(scores) 
            for domain, scores in domain_performance.items()
        }
        
        return {
            "mode": mode,
            "total_questions": len(results),
            "precision_at_k": {
                "mean": sum(precision_scores) / len(precision_scores),
                "scores": precision_scores
            },
            "recall_at_k": {
                "mean": sum(recall_scores) / len(recall_scores),
                "scores": recall_scores
            },
            "mrr": {
                "mean": sum(mrr_scores) / len(mrr_scores),
                "scores": mrr_scores
            },
            "latency": {
                "retrieval_ms": {
                    "mean": sum(retrieval_times) / len(retrieval_times) * 1000 if retrieval_times else 0,
                    "scores": [t * 1000 for t in retrieval_times]
                },
                "generation_ms": {
                    "mean": sum(generation_times) / len(generation_times) * 1000 if generation_times else 0,
                    "scores": [t * 1000 for t in generation_times]
                },
                "total_ms": {
                    "mean": sum(total_times) / len(total_times) * 1000 if total_times else 0,
                    "scores": [t * 1000 for t in total_times]
                }
            },
            "domain_performance": domain_averages,
            "keyword_match_rate": sum(len(r.keyword_matches) / len(r.expected_keywords) for r in results if r.expected_keywords) / len(results)
        }
    
    def _generate_comparison_table(self, mode_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparison table across modes."""
        
        table = {
            "precision_at_k": {},
            "recall_at_k": {},
            "mrr": {},
            "latency_total_ms": {},
            "latency_retrieval_ms": {},
            "latency_generation_ms": {}
        }
        
        for mode, results in mode_results.items():
            if "error" in results:
                continue
                
            table["precision_at_k"][mode] = results["precision_at_k"]["mean"]
            table["recall_at_k"][mode] = results["recall_at_k"]["mean"]
            table["mrr"][mode] = results["mrr"]["mean"]
            table["latency_total_ms"][mode] = results["latency"]["total_ms"]["mean"]
            table["latency_retrieval_ms"][mode] = results["latency"]["retrieval_ms"]["mean"]
            table["latency_generation_ms"][mode] = results["latency"]["generation_ms"]["mean"]
        
        return table
    
    def run_ablation_study(self, dataset_path: str) -> Dict[str, Any]:
        """Run ablation study comparing hybrid with and without reranking."""
        
        logger.info("Running ablation study: hybrid vs hybrid + reranking")
        
        dataset = self.load_dataset(dataset_path)
        hybrid_results = []
        hybrid_rerank_results = []
        
        for question_item in dataset[:10]:  # Use subset for ablation
            question = question_item["question"]
            expected_keywords = question_item["expected_keywords"]
            domain = question_item["domain"]
            
            # Hybrid without reranking
            hybrid_result = self._evaluate_single_question(
                question, expected_keywords, domain, "hybrid", 5
            )
            hybrid_results.append(hybrid_result)
            
            # Hybrid with reranking
            hybrid_rerank_result = self._evaluate_single_question(
                question, expected_keywords, domain, "hybrid_rerank", 5
            )
            hybrid_rerank_results.append(hybrid_rerank_result)
        
        # Calculate improvement
        hybrid_precision = sum(r.precision_at_k for r in hybrid_results) / len(hybrid_results)
        hybrid_rerank_precision = sum(r.precision_at_k for r in hybrid_rerank_results) / len(hybrid_rerank_results)
        
        hybrid_mrr = sum(r.mrr for r in hybrid_results) / len(hybrid_results)
        hybrid_rerank_mrr = sum(r.mrr for r in hybrid_rerank_results) / len(hybrid_rerank_results)
        
        improvement = {
            "precision_improvement": hybrid_rerank_precision - hybrid_precision,
            "mrr_improvement": hybrid_rerank_mrr - hybrid_mrr,
            "relative_precision_improvement": (hybrid_rerank_precision - hybrid_precision) / hybrid_precision * 100,
            "relative_mrr_improvement": (hybrid_rerank_mrr - hybrid_mrr) / hybrid_mrr * 100
        }
        
        return {
            "hybrid_performance": {
                "precision": hybrid_precision,
                "mrr": hybrid_mrr,
                "results": hybrid_results
            },
            "hybrid_rerank_performance": {
                "precision": hybrid_rerank_precision,
                "mrr": hybrid_rerank_mrr,
                "results": hybrid_rerank_results
            },
            "improvement": improvement
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Evaluation results saved to: %s", output_path)

# Global instance
_evaluator_instance = None

def get_evaluator() -> Evaluator:
    """Get or create evaluator instance."""
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = Evaluator()
    return _evaluator_instance
