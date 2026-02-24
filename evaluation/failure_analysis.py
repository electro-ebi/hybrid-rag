"""
Failure analysis framework for identifying system limitations.
Critical for research maturity and supervisor confidence.
"""
from __future__ import annotations

import json
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

from logger import get_logger

logger = get_logger("failure_analysis")

class FailureType(Enum):
    OCR_FAILURE = "ocr_failure"
    VISION_FAILURE = "vision_failure"
    RERANKER_MISORDER = "reranker_misorder"
    GENERATION_CONTRADICTION = "generation_contradiction"
    RETRIEVAL_FAILURE = "retrieval_failure"
    LATENCY_TIMEOUT = "latency_timeout"

@dataclass
class FailureCase:
    """Individual failure case analysis."""
    failure_type: FailureType
    question_id: str
    question: str
    expected_result: str
    actual_result: str
    confidence_score: float
    root_cause: str
    suggested_fix: str
    severity: str  # "low", "medium", "high", "critical"

class FailureAnalyzer:
    """Systematic failure analysis for research validation."""
    
    def __init__(self):
        self.failure_cases: List[FailureCase] = []
        
    def analyze_evaluation_failures(
        self, 
        evaluation_results: Dict[str, Any],
        threshold_precision: float = 0.2,
        threshold_mrr: float = 0.1,
        threshold_latency: float = 5000  # ms
    ) -> Dict[str, Any]:
        """
        Analyze evaluation results to identify failure patterns.
        
        Args:
            evaluation_results: Results from evaluation framework
            threshold_precision: Minimum acceptable precision
            threshold_mrr: Minimum acceptable MRR
            threshold_latency: Maximum acceptable latency in ms
            
        Returns:
            Comprehensive failure analysis
        """
        
        logger.info("Starting failure analysis")
        
        failure_analysis = {
            "summary": {
                "total_questions": 0,
                "failure_count": 0,
                "failure_rate": 0.0,
                "failure_types": {}
            },
            "failure_cases": [],
            "recommendations": [],
            "system_health": {}
        }
        
        # Analyze each mode
        for mode, results in evaluation_results.get("mode_results", {}).items():
            if "error" in results:
                continue
                
            mode_failures = self._analyze_mode_failures(
                mode, results, threshold_precision, threshold_mrr, threshold_latency
            )
            
            failure_analysis["failure_cases"].extend(mode_failures)
        
        # Generate summary
        total_questions = len(failure_analysis["failure_cases"])
        failure_analysis["summary"]["total_questions"] = total_questions
        failure_analysis["summary"]["failure_count"] = len(self.failure_cases)
        failure_analysis["summary"]["failure_rate"] = len(self.failure_cases) / max(total_questions, 1)
        
        # Categorize failures
        failure_types = {}
        for failure in self.failure_cases:
            failure_type = failure.failure_type.value
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
        
        failure_analysis["summary"]["failure_types"] = failure_types
        
        # Generate recommendations
        failure_analysis["recommendations"] = self._generate_recommendations()
        
        # System health assessment
        failure_analysis["system_health"] = self._assess_system_health()
        
        return failure_analysis
    
    def _analyze_mode_failures(
        self,
        mode: str,
        results: Dict[str, Any],
        threshold_precision: float,
        threshold_mrr: float,
        threshold_latency: float
    ) -> List[Dict[str, Any]]:
        """Analyze failures for a specific mode."""
        
        mode_failures = []
        
        # Check precision failures
        precision_scores = results["precision_at_k"]["scores"]
        for i, score in enumerate(precision_scores):
            if score < threshold_precision:
                failure = FailureCase(
                    failure_type=FailureType.RETRIEVAL_FAILURE,
                    question_id=f"{mode}_precision_{i}",
                    question=f"Question {i+1}",
                    expected_result=f"Precision >= {threshold_precision}",
                    actual_result=f"Precision = {score:.3f}",
                    confidence_score=score,
                    root_cause=f"Poor retrieval quality in {mode} mode",
                    suggested_fix="Improve embeddings or increase retrieval corpus",
                    severity="high" if score < 0.1 else "medium"
                )
                self.failure_cases.append(failure)
                mode_failures.append({
                    "mode": mode,
                    "failure_type": "low_precision",
                    "question_index": i,
                    "score": score,
                    "threshold": threshold_precision
                })
        
        # Check MRR failures
        mrr_scores = results["mrr"]["scores"]
        for i, score in enumerate(mrr_scores):
            if score < threshold_mrr:
                failure = FailureCase(
                    failure_type=FailureType.RERANKER_MISORDER if "rerank" in mode else FailureType.RETRIEVAL_FAILURE,
                    question_id=f"{mode}_mrr_{i}",
                    question=f"Question {i+1}",
                    expected_result=f"MRR >= {threshold_mrr}",
                    actual_result=f"MRR = {score:.3f}",
                    confidence_score=score,
                    root_cause=f"Poor ranking quality in {mode} mode",
                    suggested_fix="Improve reranking model or query understanding",
                    severity="high" if score < 0.05 else "medium"
                )
                self.failure_cases.append(failure)
                mode_failures.append({
                    "mode": mode,
                    "failure_type": "low_mrr",
                    "question_index": i,
                    "score": score,
                    "threshold": threshold_mrr
                })
        
        # Check latency failures
        latency_scores = results["latency"]["total_ms"]["scores"]
        for i, latency in enumerate(latency_scores):
            if latency > threshold_latency:
                failure = FailureCase(
                    failure_type=FailureType.LATENCY_TIMEOUT,
                    question_id=f"{mode}_latency_{i}",
                    question=f"Question {i+1}",
                    expected_result=f"Latency <= {threshold_latency}ms",
                    actual_result=f"Latency = {latency:.1f}ms",
                    confidence_score=1.0 - (latency / threshold_latency),
                    root_cause=f"Slow processing in {mode} mode",
                    suggested_fix="Optimize model inference or reduce retrieval size",
                    severity="critical" if latency > threshold_latency * 2 else "high"
                )
                self.failure_cases.append(failure)
                mode_failures.append({
                    "mode": mode,
                    "failure_type": "high_latency",
                    "question_index": i,
                    "latency": latency,
                    "threshold": threshold_latency
                })
        
        return mode_failures
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations based on failure patterns."""
        
        recommendations = []
        
        # Analyze failure patterns
        failure_counts = {}
        for failure in self.failure_cases:
            failure_type = failure.failure_type.value
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
        
        # OCR failures
        if FailureType.OCR_FAILURE.value in failure_counts:
            recommendations.append({
                "category": "OCR Improvement",
                "priority": "high",
                "description": f"{failure_counts[FailureType.OCR_FAILURE.value]} OCR failures detected",
                "actions": [
                    "Improve image preprocessing",
                    "Try multiple OCR engines",
                    "Add confidence thresholding",
                    "Implement OCR result validation"
                ]
            })
        
        # Vision failures
        if FailureType.VISION_FAILURE.value in failure_counts:
            recommendations.append({
                "category": "Vision Model Enhancement",
                "priority": "medium",
                "description": f"{failure_counts[FailureType.VISION_FAILURE.value]} vision analysis failures",
                "actions": [
                    "Fine-tune vision prompts",
                    "Add image quality assessment",
                    "Implement fallback to OCR",
                    "Use multiple vision models"
                ]
            })
        
        # Reranker failures
        if FailureType.RERANKER_MISORDER.value in failure_counts:
            recommendations.append({
                "category": "Reranking Optimization",
                "priority": "high",
                "description": f"{failure_counts[FailureType.RERANKER_MISORDER.value]} reranking failures",
                "actions": [
                    "Increase cross-encoder model size",
                    "Add query-document relevance features",
                    "Implement ensemble reranking",
                    "Add domain-specific fine-tuning"
                ]
            })
        
        # Latency issues
        if FailureType.LATENCY_TIMEOUT.value in failure_counts:
            recommendations.append({
                "category": "Performance Optimization",
                "priority": "critical",
                "description": f"{failure_counts[FailureType.LATENCY_TIMEOUT.value]} timeout failures",
                "actions": [
                    "Implement model quantization",
                    "Add request batching",
                    "Use asynchronous processing",
                    "Implement caching mechanisms"
                ]
            })
        
        # General recommendations
        recommendations.extend([
            {
                "category": "Monitoring",
                "priority": "medium",
                "description": "Add comprehensive monitoring",
                "actions": [
                    "Real-time performance metrics",
                    "Automated failure detection",
                    "Alert system for critical failures",
                    "Performance trend analysis"
                ]
            },
            {
                "category": "Testing",
                "priority": "medium",
                "description": "Expand test coverage",
                "actions": [
                    "Add domain-specific test cases",
                    "Implement stress testing",
                    "Add edge case scenarios",
                    "Continuous integration testing"
                ]
            }
        ])
        
        return recommendations
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health based on failures."""
        
        total_failures = len(self.failure_cases)
        critical_failures = sum(1 for f in self.failure_cases if f.severity == "critical")
        high_failures = sum(1 for f in self.failure_cases if f.severity == "high")
        
        # Calculate health score (0-100)
        health_score = max(0, 100 - (critical_failures * 20) - (high_failures * 10) - (total_failures * 2))
        
        # Determine health status
        if health_score >= 80:
            status = "excellent"
        elif health_score >= 60:
            status = "good"
        elif health_score >= 40:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "health_score": health_score,
            "status": status,
            "total_failures": total_failures,
            "critical_failures": critical_failures,
            "high_failures": high_failures,
            "medium_failures": sum(1 for f in self.failure_cases if f.severity == "medium"),
            "low_failures": sum(1 for f in self.failure_cases if f.severity == "low")
        }
    
    def save_failure_analysis(self, analysis: Dict[str, Any], output_path: str):
        """Save failure analysis to file."""
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info("Failure analysis saved to: %s", output_path)

# Global instance
_failure_analyzer_instance = None

def get_failure_analyzer() -> FailureAnalyzer:
    """Get or create failure analyzer instance."""
    global _failure_analyzer_instance
    if _failure_analyzer_instance is None:
        _failure_analyzer_instance = FailureAnalyzer()
    return _failure_analyzer_instance
