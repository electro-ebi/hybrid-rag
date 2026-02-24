#!/usr/bin/env python3
"""
Comprehensive research paper indexing for proper evaluation.
This script creates a robust, diverse corpus for hybrid retrieval testing.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

# Use new modular imports
from data_layer import get_sparse_store, get_vector_store
from data_layer.storage.sparse_store import count_docs
from data_layer.storage.vector_store import count_vectors
from config import WHOOSH_INDEX_DIR
from logger import get_logger

logger = get_logger("comprehensive_indexing")

def index_file(file_path: str) -> Dict[str, Any]:
    """Index a file using a simple approach."""
    try:
        # For now, let's create a simple indexing that works
        # We'll use the existing multimodal_ingest.py directly
        import sys
        from pathlib import Path
        
        # Add backend to path for imports
        backend_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(backend_dir))
        
        # Import and use the multimodal ingest
        from data_layer.multimodal_ingest import MultiModalIngestor
        
        ingestor = MultiModalIngestor()
        result = ingestor.ingest_file(file_path)
        
        return {
            "status": "success" if result.get("success", False) else "failed",
            "chunks_indexed": result.get("chunks_count", 0),
            "file_path": file_path,
            "errors": result.get("errors", {})
        }
        
    except Exception as e:
        logger.error(f"Failed to index {file_path}: {e}")
        return {
            "status": "failed",
            "chunks_indexed": 0,
            "file_path": file_path,
            "errors": {"exception": str(e)}
        }

def load_research_papers(data_dir: str) -> List[str]:
    """Load all research papers from the data directory."""
    papers = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return papers
    
    for pdf_file in data_path.glob("*.pdf"):
        papers.append(str(pdf_file))
        logger.info(f"Found paper: {pdf_file.name}")
    
    return sorted(papers)

def create_comprehensive_chunks(paper_path: str) -> List[Dict[str, Any]]:
    """Create comprehensive chunks with proper metadata."""
    # This would integrate with advanced chunking strategies
    # For now, we'll use the standard indexing but ensure proper metadata
    return []

def index_all_papers(data_dir: str = "../data") -> Dict[str, Any]:
    """Index all research papers with comprehensive processing."""
    logger.info("Starting comprehensive research paper indexing")
    
    # Load all papers
    papers = load_research_papers(data_dir)
    if not papers:
        logger.error("No papers found to index")
        return {"status": "error", "message": "No papers found"}
    
    logger.info(f"Found {len(papers)} papers to index")
    
    # Index each paper
    indexing_results = {
        "status": "success",
        "total_papers": len(papers),
        "successful": 0,
        "failed": 0,
        "total_chunks": 0,
        "papers": []
    }
    
    for paper_path in papers:
        paper_name = os.path.basename(paper_path)
        logger.info(f"Indexing: {paper_name}")
        
        try:
            result = index_file(paper_path)
            
            if result.get("status") == "success":
                chunks = result.get("chunks_indexed", 0)
                indexing_results["successful"] += 1
                indexing_results["total_chunks"] += chunks
                
                paper_info = {
                    "name": paper_name,
                    "status": "success",
                    "chunks": chunks,
                    "sparse_indexed": result.get("sparse_indexed", False),
                    "dense_indexed": result.get("dense_indexed", False)
                }
                indexing_results["papers"].append(paper_info)
                
                logger.info(f"✅ {paper_name}: {chunks} chunks indexed")
            else:
                indexing_results["failed"] += 1
                error_msg = result.get("errors", "Unknown error")
                logger.error(f"❌ {paper_name}: {error_msg}")
                
                paper_info = {
                    "name": paper_name,
                    "status": "failed",
                    "error": error_msg
                }
                indexing_results["papers"].append(paper_info)
                
        except Exception as e:
            indexing_results["failed"] += 1
            logger.error(f"❌ {paper_name}: Exception - {e}")
            
            paper_info = {
                "name": paper_name,
                "status": "failed",
                "error": str(e)
            }
            indexing_results["papers"].append(paper_info)
    
    # Verify final counts
    final_sparse = count_docs(WHOOSH_INDEX_DIR)
    final_dense = count_vectors()
    
    indexing_results["final_counts"] = {
        "sparse_documents": final_sparse,
        "vector_documents": final_dense
    }
    
    logger.info(f"Indexing complete: {indexing_results['successful']}/{len(papers)} successful")
    logger.info(f"Total chunks: {indexing_results['total_chunks']}")
    logger.info(f"Final counts: Sparse={final_sparse}, Dense={final_dense}")
    
    return indexing_results

def create_corpus_metadata(indexing_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create comprehensive metadata about the indexed corpus."""
    metadata = {
        "corpus_info": {
            "total_papers": indexing_results["total_papers"],
            "successful_papers": indexing_results["successful"],
            "total_chunks": indexing_results["total_chunks"],
            "creation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "indexing_version": "1.0"
        },
        "paper_details": indexing_results["papers"],
        "final_counts": indexing_results["final_counts"],
        "domains_covered": [
            "retrieval augmented generation",
            "hybrid search systems", 
            "vector databases",
            "cross-encoder reranking",
            "contextual chunking",
            "information retrieval",
            "embedding models",
            "natural language processing"
        ]
    }
    
    return metadata

def main():
    """Main indexing function."""
    print("🚀 COMPREHENSIVE RESEARCH PAPER INDEXING")
    print("=" * 50)
    
    # Index all papers
    results = index_all_papers()
    
    if results["status"] == "success":
        # Create metadata
        metadata = create_corpus_metadata(results)
        
        # Save metadata
        metadata_path = "evaluation/corpus_metadata.json"
        os.makedirs("evaluation", exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ INDEXING SUCCESSFUL!")
        print(f"📚 Papers indexed: {results['successful']}/{results['total_papers']}")
        print(f"📄 Total chunks: {results['total_chunks']}")
        print(f"📊 Sparse documents: {results['final_counts']['sparse_documents']}")
        print(f"🔢 Vector documents: {results['final_counts']['vector_documents']}")
        print(f"💾 Metadata saved: {metadata_path}")
        
        # Assessment
        total_chunks = results['total_chunks']
        if total_chunks >= 500:
            print(f"🎯 CORPUS SIZE: {total_chunks} chunks - Suitable for evaluation")
        else:
            print(f"⚠️  CORPUS SIZE: {total_chunks} chunks - Consider adding more papers")
            
    else:
        print(f"\n❌ INDEXING FAILED!")
        print(f"Error: {results.get('message', 'Unknown error')}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
