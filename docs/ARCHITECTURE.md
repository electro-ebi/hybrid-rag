# System Architecture Documentation

## 🏗️ Clean Architecture Overview

### 📁 Core Production Files
```
main.py                    - Main application entry point
config.py                  - System configuration
logger.py                  - Logging system
rag_pipeline.py           - MAIN RAG orchestration (single entry point)
llm.py                     - LLM integration
model_router.py            - Model routing
vision_handler.py          - Vision processing
```

### 📂 Data Layer Structure
```
data_layer/
├── core/                   - Base classes and configuration
├── storage/                - Sparse (Whoosh) + Dense (ChromaDB) storage
├── retrieval/              - Dense, Sparse, Hybrid retrievers
├── processing/             - Text, PDF, OCR, Vision processing
├── indexing/               - Document indexing
├── loaders/                - File loaders (PDF, text, code)
└── utils/                  - Utilities and validation
```

## 🔄 Retrieval Pipeline (Single Clean Flow)

```
rag_pipeline.py
    ↓
query_processor.py
    ↓
├── sparse_retriever.py (Whoosh)
└── dense_retriever.py (ChromaDB)
    ↓
fusion.py
    ↓
cross_encoder_reranker.py (optional)
    ↓
Final Results
```

## 📊 Index Status
- **Sparse Index (Whoosh)**: Full document corpus
- **Vector Index (ChromaDB)**: Symmetric document corpus
- **Status**: Synchronized and symmetric

## 🎯 Key Design Decisions

1. **Single Entry Point**: All RAG operations through rag_pipeline.py
2. **Symmetric Indexing**: Sparse and dense indices contain same documents
3. **Clean Separation**: Each module has single responsibility
4. **No Legacy Code**: All legacy files moved to archive/
5. **Production Ready**: No experimental features in main codebase

## 📈 Performance Characteristics
- Sparse Retrieval: ~5ms
- Dense Retrieval: ~85ms  
- Hybrid Retrieval: ~90ms
- Neural Reranking: +750ms

## 🚀 Usage Examples

### Basic Query
```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
results = pipeline.query("hybrid retrieval systems", top_k=10)
```

### Advanced Query with Reranking
```python
results = pipeline.query(
    "semantic search algorithms", 
    top_k=10, 
    use_reranking=True
)
```

## 📝 Maintenance Notes
- Archive directory contains all legacy code
- System status tracked in system_status.json
- Configuration managed through stable_config.json
