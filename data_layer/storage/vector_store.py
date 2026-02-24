"""
Vector storage implementation using ChromaDB.
"""

from typing import Any, List, Dict, Optional
import chromadb
from chromadb.config import Settings

from ..core.base import BaseStorage
from ..core.config import DEFAULT_CONFIG


class VectorStore(BaseStorage):
    """ChromaDB-based vector storage for semantic search."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize vector store."""
        self.config = DEFAULT_CONFIG
        if config:
            self.config = self.config.from_dict(config)
        
        self.client = chromadb.PersistentClient(
            path=self.config.vector_index_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def store(self, key: str, value: Any) -> bool:
        """Store a chunk in vector store."""
        try:
            if isinstance(value, dict) and 'content' in value:
                # Legacy dict format
                content = value['content']
                metadata = {k: v for k, v in value.items() if k != 'content'}
            else:
                # New format
                content = str(value)
                metadata = {}
            
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[key]
            )
            return True
        except Exception as e:
            print(f"Error storing in vector store: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a chunk by ID."""
        try:
            result = self.collection.get(ids=[key])
            if result['documents']:
                return {
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0] if result['metadatas'] else {}
                }
            return None
        except Exception as e:
            print(f"Error retrieving from vector store: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a chunk by ID."""
        try:
            self.collection.delete(ids=[key])
            return True
        except Exception as e:
            print(f"Error deleting from vector store: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if chunk exists."""
        return self.retrieve(key) is not None
    
    def list_keys(self, pattern: str = "*") -> List[str]:
        """List all chunk IDs."""
        try:
            result = self.collection.get()
            return result['ids']
        except Exception as e:
            print(f"Error listing keys from vector store: {e}")
            return []
    
    def query(self, query_text: str, top_k: int = 10) -> Dict[str, Any]:
        """Query for similar chunks."""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k
            )
            
            return {
                'documents': results['documents'][0] if results['documents'] else [],
                'ids': results['ids'][0] if results['ids'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'distances': results['distances'][0] if results['distances'] else []
            }
        except Exception as e:
            print(f"Error querying vector store: {e}")
            return {'documents': [], 'ids': [], 'metadatas': [], 'distances': []}
    
    def count_vectors(self) -> int:
        """Count total vectors in store."""
        try:
            result = self.collection.get()
            return len(result['ids'])
        except Exception:
            return 0


# Legacy compatibility functions
def query_similar(query_embedding: List[float], top_k: int = 10) -> Dict[str, Any]:
    """Legacy query function (note: this needs embedding, not text)."""
    # This is a simplified version - in practice, you'd need to convert embedding to text
    store = VectorStore()
    # For now, return empty result - this needs proper implementation
    return {'documents': [], 'ids': [], 'metadatas': [], 'distances': []}


def add_chunks(chunks: List[Dict[str, Any]]) -> bool:
    """Legacy add chunks function."""
    store = VectorStore()
    success = True
    
    for chunk in chunks:
        chunk_id = chunk.get('chunk_uid', chunk.get('id', ''))
        if chunk_id:
            if not store.store(chunk_id, chunk):
                success = False
    
    return success


def count_vectors() -> int:
    """Legacy count function."""
    store = VectorStore()
    return store.count_vectors()
