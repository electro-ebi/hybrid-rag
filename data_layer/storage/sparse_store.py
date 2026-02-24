"""
Sparse storage implementation using Whoosh.
"""

from __future__ import annotations

import os
from typing import Any, List, Dict, Optional

from whoosh import index
from whoosh.fields import ID, STORED, TEXT, Schema
from whoosh.qparser import MultifieldParser
from whoosh.query import Term
from whoosh.analysis import StandardAnalyzer

from ..core.base import BaseStorage
from ..core.config import DEFAULT_CONFIG
from ..core.types import Chunk


class SparseStore(BaseStorage):
    """Whoosh-based sparse storage for keyword search."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize sparse store."""
        self.config = DEFAULT_CONFIG
        if config:
            self.config = self.config.from_dict(config)
        
        self.index_dir = self.config.sparse_index_dir
        self._index = None
    
    def _get_schema(self) -> Schema:
        """Create schema for sparse index."""
        analyzer = StandardAnalyzer(stoplist=self.config.sparse_stoplist)
        return Schema(
            chunk_uid=ID(stored=True, unique=True),
            source=STORED,
            file_type=STORED,
            file_hash=ID(stored=True),
            chunk_id=STORED,
            content=TEXT(stored=True, analyzer=analyzer),
        )
    
    def _get_index(self):
        """Get or create Whoosh index."""
        if self._index is None:
            os.makedirs(self.index_dir, exist_ok=True)
            if index.exists_in(self.index_dir):
                self._index = index.open_dir(self.index_dir)
            else:
                self._index = index.create_in(self.index_dir, self._get_schema())
        return self._index
    
    def store(self, key: str, value: Any) -> bool:
        """Store a chunk in sparse index."""
        try:
            ix = self._get_index()
            writer = ix.writer()
            
            if isinstance(value, Chunk):
                writer.add_document(
                    chunk_uid=value.id,
                    source=value.source,
                    file_type=value.metadata.get('file_type', ''),
                    file_hash=value.metadata.get('file_hash', ''),
                    chunk_id=value.metadata.get('chunk_id', ''),
                    content=value.content
                )
            else:
                # Legacy support for dict format
                writer.add_document(
                    chunk_uid=value.get('chunk_uid', key),
                    source=value.get('source', ''),
                    file_type=value.get('file_type', ''),
                    file_hash=value.get('file_hash', ''),
                    chunk_id=value.get('chunk_id', ''),
                    content=value.get('content', '')
                )
            
            writer.commit()
            return True
        except Exception as e:
            print(f"Error storing in sparse index: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a chunk by ID."""
        try:
            ix = self._get_index()
            with ix.searcher() as searcher:
                doc = searcher.document(chunk_uid=key)
                return doc if doc else None
        except Exception as e:
            print(f"Error retrieving from sparse index: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a chunk by ID."""
        try:
            ix = self._get_index()
            writer = ix.writer()
            writer.delete_by_term('chunk_uid', key)
            writer.commit()
            return True
        except Exception as e:
            print(f"Error deleting from sparse index: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if chunk exists."""
        return self.retrieve(key) is not None
    
    def list_keys(self, pattern: str = "*") -> List[str]:
        """List all chunk IDs."""
        try:
            ix = self._get_index()
            with ix.searcher() as searcher:
                reader = searcher.reader()
                return list(reader.field_terms('chunk_uid'))
        except Exception as e:
            print(f"Error listing keys from sparse index: {e}")
            return []
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for chunks matching query."""
        try:
            ix = self._get_index()
            with ix.searcher() as searcher:
                parser = MultifieldParser(['content'], ix.schema)
                query_obj = parser.parse(query)
                results = searcher.search(query_obj, limit=top_k)
                
                return [
                    {
                        'chunk_uid': hit['chunk_uid'],
                        'content': hit['content'],
                        'source': hit['source'],
                        'score': hit.score,
                        'file_type': hit.get('file_type', ''),
                        'chunk_id': hit.get('chunk_id', '')
                    }
                    for hit in results
                ]
        except Exception as e:
            print(f"Error searching sparse index: {e}")
            return []
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add multiple chunks to index."""
        try:
            ix = self._get_index()
            writer = ix.writer()
            
            for chunk in chunks:
                writer.add_document(
                    chunk_uid=chunk['chunk_uid'],
                    source=chunk['source'],
                    file_type=chunk.get('file_type', ''),
                    file_hash=chunk.get('file_hash', ''),
                    chunk_id=chunk.get('chunk_id', ''),
                    content=chunk['content']
                )
            
            writer.commit()
            return True
        except Exception as e:
            print(f"Error adding chunks to sparse index: {e}")
            return False
    
    def count_documents(self) -> int:
        """Count total documents in index."""
        try:
            ix = self._get_index()
            with ix.searcher() as searcher:
                return searcher.doc_count()
        except Exception:
            return 0


# Legacy compatibility functions
def _schema() -> Schema:
    """Legacy schema function."""
    store = SparseStore()
    return store._get_schema()


def get_or_create_index(index_dir: str):
    """Legacy index creation function."""
    config = DEFAULT_CONFIG
    config.sparse_index_dir = index_dir
    store = SparseStore(config.to_dict())
    return store._get_index()


def search(index_dir: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Legacy search function."""
    config = DEFAULT_CONFIG
    config.sparse_index_dir = index_dir
    store = SparseStore(config.to_dict())
    return store.search(query, top_k)


def add_sparse_chunks(chunks: List[Dict[str, Any]], index_dir: str) -> bool:
    """Legacy add chunks function."""
    config = DEFAULT_CONFIG
    config.sparse_index_dir = index_dir
    store = SparseStore(config.to_dict())
    return store.add_chunks(chunks)


def count_docs(index_dir: str) -> int:
    """Legacy count function."""
    config = DEFAULT_CONFIG
    config.sparse_index_dir = index_dir
    store = SparseStore(config.to_dict())
    return store.count_documents()
