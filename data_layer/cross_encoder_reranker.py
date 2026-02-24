"""
Production-ready cross-encoder reranker using BAAI/bge-reranker-base.
Optimized for RTX 4060 (8GB) - runs on CPU to preserve GPU VRAM.
"""
from __future__ import annotations

import os
import time
from typing import List, Tuple, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from logger import get_logger

logger = get_logger("cross_encoder_reranker")

# Model configuration
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_MAX_CHUNKS = 20  # Increased for better results
RERANK_BATCH_SIZE = 4   # Optimize for CPU performance

class CrossEncoderReranker:
    """Enterprise-grade cross-encoder reranker."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize reranker with configuration."""
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.device = self._detect_device()
        self._load_model()
    
    def _detect_device(self) -> str:
        """Auto-detect optimal device. Prefer CPU for reranker to save GPU VRAM."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if gpu_memory >= 6:  # RTX 4060 has 8GB
                logger.info("GPU detected (%.1fGB), but using CPU for reranker to preserve VRAM", gpu_memory)
                return "cpu"
            else:
                logger.info("GPU detected with low memory (%.1fGB), using CPU", gpu_memory)
                return "cpu"
        else:
            logger.info("No GPU detected, using CPU")
            return "cpu"
    
    def _load_model(self):
        """Load BAAI/bge-reranker-base model and tokenizer."""
        try:
            model_name = self.config.get('reranker_model', RERANKER_MODEL)
            logger.info("Loading cross-encoder reranker: %s", model_name)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Cross-encoder reranker loaded on device: %s", self.device)
            
        except Exception as e:
            logger.error("Failed to load cross-encoder reranker: %s", e)
            raise
    
    def rerank(
        self, 
        query: str, 
        chunks: List[str], 
        chunk_ids: List[str] | None = None
    ) -> List[str]:
        """
        Rerank chunks using cross-encoder scoring.
        
        Args:
            query: Search query
            chunks: List of chunk texts
            chunk_ids: Optional list of chunk IDs (not used in ordering)
            
        Returns:
            List of chunks sorted by relevance (descending)
        """
        if not chunks:
            return []
        
        if chunk_ids is None:
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Limit chunks for performance
        max_chunks = self.config.get('reranker_max_chunks', RERANK_MAX_CHUNKS)
        chunks_to_rerank = chunks[:max_chunks]
        overflow_chunks = chunks[max_chunks:]
        
        start_time = time.perf_counter()
        
        try:
            # Prepare inputs for cross-encoder
            inputs = []
            for chunk in chunks_to_rerank:
                # Truncate chunk to avoid token limit issues
                truncated_chunk = chunk[:2000]  # Reasonable limit
                inputs.append((query, truncated_chunk))
            
            # Batch processing for efficiency
            scores = self._batch_score(inputs)
            
            # Pair scores with chunks and sort
            scored_chunks = list(zip(scores, chunks_to_rerank))
            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            
            # Extract sorted chunks
            reranked_chunks = [chunk for _, chunk in scored_chunks]
            
            # Add overflow chunks (maintain original order)
            reranked_chunks.extend(overflow_chunks)
            
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Cross-encoder reranking completed in %.2fs for %d chunks (device: %s)",
                elapsed, len(chunks_to_rerank), self.device
            )
            
            return reranked_chunks
            
        except Exception as e:
            logger.error("Cross-encoder reranking failed: %s", e)
            logger.info("Falling back to original chunk order")
            return chunks
    
    def _batch_score(self, inputs: List[Tuple[str, str]]) -> List[float]:
        """Score query-chunk pairs in batches."""
        scores = []
        batch_size = self.config.get('reranker_batch_size', RERANK_BATCH_SIZE)
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            # Tokenize batch
            batch_queries = [pair[0] for pair in batch]
            batch_chunks = [pair[1] for pair in batch]
            
            # BGE reranker expects: [query, passage] pairs
            tokenized = self.tokenizer(
                batch_queries,
                batch_chunks,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            
            # Get scores
            with torch.no_grad():
                outputs = self.model(**tokenized)
                batch_scores = outputs.logits.squeeze(-1).cpu().tolist()
                
                # Convert to float if single item
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                
                scores.extend(batch_scores)
        
        return scores
    
    def is_available(self) -> bool:
        """Check if reranker is available."""
        return self.model is not None and self.tokenizer is not None


# Global instance for reuse
_reranker_instance = None

def get_reranker() -> CrossEncoderReranker:
    """Get or create global reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance

def rerank_chunks(query: str, chunks: List[str], chunk_ids: List[str] | None = None) -> List[str]:
    """
    Convenience function for reranking chunks.
    
    Args:
        query: Search query
        chunks: List of chunk texts to rerank
        chunk_ids: Optional chunk IDs
        
    Returns:
        Reranked chunks (most relevant first)
    """
    reranker = get_reranker()
    return reranker.rerank(query, chunks, chunk_ids)
