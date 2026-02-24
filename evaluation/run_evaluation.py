#!/usr/bin/env python3
"""
Run comprehensive evaluation of the autonomous AI agent.
Produces tables and metrics for research validation.
"""
import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluator import get_evaluator
from logger import get_logger

logger = get_logger("run_evaluation")

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation")
    parser.add_argument("--dataset", default="evaluation/dataset.json", help="Dataset path")
    parser.add_argument("--output", default="evaluation/results", help="Output directory")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--sample", type=int, help="Use sample of N questions")
    parser.add_argument("--retrieval-only", action="store_true", help="Run retrieval-only evaluation (no LLM)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize evaluator
    evaluator = get_evaluator()
    
    if args.retrieval_only:
        print("🧪 Starting RETRIEVAL-ONLY Evaluation (No LLM Bottleneck)")
        print(f"📊 Dataset: {args.dataset}")
        print(f"🎯 Top-K: {args.top_k}")
        print(f"📁 Output: {args.output}")
        
        # Run retrieval-only evaluation
        print("\n🔥 Running Retrieval-Only Evaluation...")
        results = evaluator.evaluate_retrieval_only(
            dataset_path=args.dataset,
            top_k=args.top_k,
            modes=["dense", "sparse", "hybrid", "hybrid_rerank"]
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_output = f"{args.output}/retrieval_only_{timestamp}.json"
        evaluator.save_results(results, main_output)
        
        # Print comparison table
        print("\n📊 RETRIEVAL COMPARISON TABLE")
        print("=" * 80)
        comparison = results["comparison_table"]
        
        print(f"{'Mode':<15} {'Precision@5':<12} {'Recall@5':<10} {'MRR':<8} {'Retrieval (ms)':<15}")
        print("-" * 80)
        
        for mode in ["dense", "sparse", "hybrid", "hybrid_rerank"]:
            if mode in comparison["precision_at_k"]:
                precision = comparison["precision_at_k"][mode]
                recall = comparison["recall_at_k"][mode]
                mrr = comparison["mrr"][mode]
                retrieval_latency = comparison["latency_retrieval_ms"][mode]
                
                print(f"{mode:<15} {precision:<12.3f} {recall:<10.3f} {mrr:<8.3f} {retrieval_latency:<15.1f}")
        
        print(f"\n✅ Retrieval-Only Evaluation Complete!")
        print(f"📁 Results saved to: {args.output}/")
        print(f"🕐 Timestamp: {timestamp}")
        
    else:
        print("🧪 Starting Full Pipeline Evaluation")
        print(f"📊 Dataset: {args.dataset}")
        print(f"🎯 Top-K: {args.top_k}")
        print(f"📁 Output: {args.output}")
        
        # Run main evaluation
        print("\n🔥 Running Full Pipeline Evaluation...")
        results = evaluator.evaluate_full_pipeline(
            dataset_path=args.dataset,
            top_k=args.top_k,
            modes=["dense", "sparse", "hybrid", "hybrid_rerank"]
        )
        
        # Save main results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_output = f"{args.output}/main_evaluation_{timestamp}.json"
        evaluator.save_results(results, main_output)
        
        # Print comparison table
        print("\n📊 COMPARISON TABLE")
        print("=" * 80)
        comparison = results["comparison_table"]
        
        print(f"{'Mode':<15} {'Precision@5':<12} {'Recall@5':<10} {'MRR':<8} {'Total Latency (ms)':<18}")
        print("-" * 80)
        
        for mode in ["dense", "sparse", "hybrid", "hybrid_rerank"]:
            if mode in comparison["precision_at_k"]:
                precision = comparison["precision_at_k"][mode]
                recall = comparison["recall_at_k"][mode]
                mrr = comparison["mrr"][mode]
                latency = comparison["latency_total_ms"][mode]
                
                print(f"{mode:<15} {precision:<12.3f} {recall:<10.3f} {mrr:<8.3f} {latency:<18.1f}")
        
        # Print latency breakdown
        print(f"\n⚡ LATENCY BREAKDOWN (ms)")
        print("=" * 60)
        print(f"{'Mode':<15} {'Retrieval':<10} {'Generation':<12} {'Total':<8}")
        print("-" * 60)
        
        for mode in ["dense", "sparse", "hybrid", "hybrid_rerank"]:
            if mode in comparison["latency_retrieval_ms"]:
                retrieval = comparison["latency_retrieval_ms"][mode]
                generation = comparison["latency_generation_ms"][mode]
                total = comparison["latency_total_ms"][mode]
                
                print(f"{mode:<15} {retrieval:<10.1f} {generation:<12.1f} {total:<8.1f}")
        
        # Run ablation study if requested
        if args.ablation:
            print(f"\n🔬 Running Ablation Study...")
            ablation_results = evaluator.run_ablation_study(args.dataset)
            
            ablation_output = f"{args.output}/ablation_study_{timestamp}.json"
            evaluator.save_results(ablation_results, ablation_output)
            
            print("\n📈 ABLATION RESULTS")
            print("=" * 50)
            improvement = ablation_results["improvement"]
            
            print(f"Hybrid Precision: {ablation_results['hybrid_performance']['precision']:.3f}")
            print(f"Hybrid + Rerank Precision: {ablation_results['hybrid_rerank_performance']['precision']:.3f}")
            print(f"Precision Improvement: {improvement['precision_improvement']:.3f} ({improvement['relative_precision_improvement']:.1f}%)")
            
            print(f"\nHybrid MRR: {ablation_results['hybrid_performance']['mrr']:.3f}")
            print(f"Hybrid + Rerank MRR: {ablation_results['hybrid_rerank_performance']['mrr']:.3f}")
            print(f"MRR Improvement: {improvement['mrr_improvement']:.3f} ({improvement['relative_mrr_improvement']:.1f}%)")
        
        # Generate summary report
        print(f"\n📋 SUMMARY REPORT")
        print("=" * 40)
        print(f"Dataset Size: {results['dataset_size']} questions")
        print(f"Modes Evaluated: {', '.join(results['evaluation_config']['modes'])}")
        
        # Find best performing mode
        best_precision = max(comparison["precision_at_k"].values())
        best_mode = max(comparison["precision_at_k"], key=comparison["precision_at_k"].get)
        
        print(f"Best Precision@5: {best_precision:.3f} ({best_mode})")
        
        best_mrr = max(comparison["mrr"].values())
        best_mrr_mode = max(comparison["mrr"], key=comparison["mrr"].get)
        
        print(f"Best MRR: {best_mrr:.3f} ({best_mrr_mode})")
        
        print(f"\n✅ Evaluation Complete!")
        print(f"📁 Results saved to: {args.output}/")
        print(f"🕐 Timestamp: {timestamp}")

if __name__ == "__main__":
    main()
