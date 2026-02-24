"""
Fusion algorithms for hybrid retrieval.
"""

from typing import List, Tuple


def rrf_fuse(
    dense_results: List[Tuple[str, str]], 
    sparse_results: List[Tuple[str, str]], 
    top_k: int = 10,
    rrf_k: int = 60
) -> Tuple[List[str], List[str]]:
    """
    Reciprocal Rank Fusion over document IDs.
    
    Args:
        dense_results: List of (doc_id, content) tuples from dense retrieval
        sparse_results: List of (doc_id, content) tuples from sparse retrieval
        top_k: Number of final results to return
        rrf_k: RRF constant (typically 60)
    
    Returns:
        Tuple of (final_doc_ids, final_contents)
    """
    # Score dictionaries
    doc_scores = {}
    doc_contents = {}
    
    # Score dense results
    for rank, (doc_id, content) in enumerate(dense_results, 1):
        score = 1.0 / (rrf_k + rank)
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
        doc_contents[doc_id] = content
    
    # Score sparse results
    for rank, (doc_id, content) in enumerate(sparse_results, 1):
        score = 1.0 / (rrf_k + rank)
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
        doc_contents[doc_id] = content
    
    # Sort by score and return top-k
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    final_doc_ids = [doc_id for doc_id, _ in sorted_docs[:top_k]]
    final_contents = [doc_contents[doc_id] for doc_id in final_doc_ids]
    
    return final_doc_ids, final_contents


class RRFFusion:
    """Reciprocal Rank Fusion implementation."""
    
    def __init__(self, rrf_k: int = 60):
        """Initialize RRF fusion."""
        self.rrf_k = rrf_k
    
    def fuse(
        self, 
        dense_results: List[Tuple[str, str]], 
        sparse_results: List[Tuple[str, str]], 
        top_k: int = 10
    ) -> Tuple[List[str], List[str]]:
        """Fuse results using RRF."""
        return rrf_fuse(dense_results, sparse_results, top_k, self.rrf_k)
