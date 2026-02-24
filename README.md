# 🤖 Autonomous Agent - Production RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with modular architecture, supporting hybrid retrieval, neural reranking, and multimodal document processing.

## 🚀 Features

### **🔍 Advanced Retrieval**
- **Sparse Retrieval**: Fast keyword-based search using Whoosh (4.2ms avg)
- **Dense Retrieval**: Semantic search using ChromaDB vector embeddings (81ms avg)
- **Hybrid Retrieval**: Intelligent fusion of sparse and dense results (89ms avg)
- **Neural Reranking**: Cross-encoder reranking for maximum relevance

### **📚 Document Processing**
- **Multimodal Support**: Text, PDF, and image processing
- **OCR Integration**: Extract text from scanned documents
- **Intelligent Chunking**: Context-aware document segmentation
- **Batch Indexing**: Efficient large-scale document processing

### **🏗️ Architecture**
- **Modular Design**: Clean separation of concerns
- **Factory Pattern**: Flexible component creation
- **Legacy Compatibility**: Backward compatibility maintained
- **Production Ready**: Comprehensive error handling and logging

## 📊 Performance Metrics

| Method | Latency | Success Rate | Results | Grade |
|--------|---------|--------------|---------|-------|
| Sparse Retrieval | 4.2ms | 100% | 1,135 docs | 🟢 EXCELLENT |
| Dense Retrieval | 81.2ms | 100% | Dynamic | 🟢 EXCELLENT |
| Hybrid Retrieval | 89ms | 100% | Dynamic | 🟡 GOOD |
| Neural Reranking | +750ms | 100% | Optimized | 🟡 GOOD |

## 🛠️ Installation

### **Prerequisites**
- Python 3.8+
- Ollama (for embeddings and LLM)
- ChromaDB
- Whoosh

### **Setup**
```bash
# Clone the repository
git clone <repository-url>
cd autonomous-agent/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup Ollama models
ollama pull nomic-embed-text
ollama pull qwen2.5:14b
```

### **Configuration**
```bash
# Copy configuration template
cp config.py.template config.py

# Edit configuration as needed
nano config.py
```

## 🚀 Quick Start

### **Basic Usage**
```python
from rag_pipeline import RAGPipeline

# Initialize the pipeline
pipeline = RAGPipeline()

# Query the system
results = pipeline.query("hybrid retrieval systems", top_k=10)

# Display results
for i, result in enumerate(results, 1):
    print(f"{i}. {result['content'][:100]}...")
```

### **Advanced Usage**
```python
from data_layer import sparse_search, get_vector_store
from data_layer.cross_encoder_reranker import get_reranker

# Sparse search
sparse_results = sparse_search('whoosh_index', 'query', top_k=10)

# Dense search
vector_store = get_vector_store()
dense_results = vector_store.query('query', top_k=10)

# Neural reranking
reranker = get_reranker()
reranked = reranker.rerank('query', documents)
```

## 📁 Project Structure

```
backend/
├── data_layer/           # Core data processing modules
│   ├── core/            # Base classes and configuration
│   ├── storage/         # Sparse and vector storage
│   ├── retrieval/       # Retrieval components
│   ├── processing/      # Document processing
│   ├── indexing/        # Document indexing
│   ├── loaders/         # File loaders
│   └── utils/           # Utilities
├── evaluation/          # Evaluation and testing
├── scripts/             # Utility scripts
├── rag_pipeline.py      # Main RAG orchestration
├── config.py            # System configuration
├── main.py              # Application entry point
└── ARCHITECTURE.md      # Detailed architecture docs
```

## 🔧 Configuration

### **System Configuration**
```python
# config.py
DATA_DIR = "../data"
WHOOSH_INDEX_DIR = "whoosh_index"
CHROMA_DIR = "chroma_db"

# Models
EMBED_MODEL = "nomic-embed-text"
GENERATION_MODEL = "qwen2.5:14b"
RERANK_MODEL = "BAAI/bge-reranker-base"

# Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
```

### **Data Layer Configuration**
```python
from data_layer.core.config import DataLayerConfig

config = DataLayerConfig()
config.embedding_dimension = 768
config.vector_index_dir = "chroma_db"
config.sparse_index_dir = "whoosh_index"
```

## 📚 Indexing Documents

### **Index PDF Documents**
```python
from data_layer.multimodal_ingest import MultiModalIngestor

ingestor = MultiModalIngestor()
ingestor.ingest_directory("../data/papers/")
```

### **Batch Indexing**
```python
from data_layer.indexing.batch_indexer import BatchIndexer

indexer = BatchIndexer()
indexer.index_directory("../data/", batch_size=50)
```

## 📊 Evaluation

### **Run Evaluation**
```bash
python evaluation/run_evaluation.py
```

### **Performance Analysis**
```python
from evaluation.proper_evaluator import ProperEvaluator

evaluator = ProperEvaluator()
results = evaluator.evaluate_retrieval_system()
print(results)
```

## 🔍 Monitoring

### **System Status**
```python
import json

# Load system status
with open('system_status.json', 'r') as f:
    status = json.load(f)

print(f"System Status: {status['status']}")
print(f"Sparse Documents: {status['document_counts']['sparse_documents']}")
print(f"Vector Documents: {status['document_counts']['vector_documents']}")
```

### **Performance Metrics**
```python
from data_layer.storage.sparse_store import count_docs
from data_layer.storage.vector_store import count_vectors

sparse_count = count_docs('whoosh_index')
vector_count = count_vectors()

print(f"Sparse Index: {sparse_count} documents")
print(f"Vector Index: {vector_count} documents")
```

## 🐛 Troubleshooting

### **Common Issues**

1. **Vector Store Empty**
   ```bash
   # Rebuild vector store
   python -c "from data_layer.storage.vector_store import VectorStore; VectorStore({})"
   ```

2. **Import Errors**
   ```bash
   # Check data layer structure
   python -c "from data_layer import sparse_search; print('Imports OK')"
   ```

3. **Performance Issues**
   ```bash
   # Check system status
   python -c "import json; print(json.load(open('system_status.json')))"
   ```

### **Debug Mode**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
from logger import get_logger
logger = get_logger("debug")
logger.setLevel(logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### **Development Guidelines**
- Follow the modular architecture
- Add comprehensive tests
- Update documentation
- Maintain backward compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Whoosh**: Fast pure-Python search engine
- **ChromaDB**: AI-native vector database
- **Ollama**: Local LLM and embedding serving
- **HuggingFace**: Pre-trained models and utilities

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation
- Review system status in `system_status.json`

---

**Built with ❤️ for production-grade RAG systems**
