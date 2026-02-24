"""
Retrieval layer for hybrid search.
"""

from .query_processor import QueryProcessor
from .fusion import RRFFusion

# Legacy compatibility - simple approach
def retrieve_chunks(*args, **kwargs):
    """Legacy retrieve_chunks function - redirects to hybrid retriever."""
    # For now, use a simple implementation
    from ..storage.sparse_store import search as sparse_search
    from ..storage.vector_store import query_similar
    from ..cross_encoder_reranker import rerank_chunks
    
    question = args[0] if args else kwargs.get('question', '')
    top_k = kwargs.get('top_k', 5)
    mode = kwargs.get('mode', 'hybrid')
    
    if mode == 'sparse':
        results = sparse_search('whoosh_index', question, top_k)
        return [r['content'] for r in results]
    elif mode == 'dense':
        # This would need embedding generation - simplified for now
        return []
    else:
        # Hybrid - combine both
        sparse_results = sparse_search('whoosh_index', question, top_k)
        return [r['content'] for r in sparse_results]

__all__ = [
    'QueryProcessor',
    'RRFFusion',
    'retrieve_chunks'
]
