"""
Main RAG Pipeline - Single Entry Point
=====================================

This is the primary orchestration file for the RAG system.
All retrieval operations should go through this pipeline.

Usage:
    from rag_pipeline import RAGPipeline
    
    pipeline = RAGPipeline()
    results = pipeline.query("your query", top_k=10)

Architecture:
    1. Query Processing
    2. Sparse Retrieval (Whoosh)
    3. Dense Retrieval (ChromaDB) 
    4. Result Fusion
    5. Neural Reranking (optional)

Note: This replaces the old rag.py file for clarity.
"""

from data_layer.embeddings import generate_embedding
from data_layer.vector_store import query_similar
from llm import call_llm


def format_context(chunks: list) -> str:
    """
    Format retrieved chunks into structured context.
    """

    formatted = ""
    for i, chunk in enumerate(chunks):
        formatted += f"\n\n[Chunk {i+1}]\n{chunk}"

    return formatted


def answer_query(question: str, top_k: int = 3) -> dict:
    """
    Full RAG pipeline:
    embed → retrieve → format → generate answer
    """

    # Step 1: Embed question
    query_embedding = generate_embedding(question)

    # Step 2: Retrieve similar chunks
    results = query_similar(query_embedding, top_k=top_k)

    retrieved_chunks = results["documents"][0]

    # Step 3: Format context
    context = format_context(retrieved_chunks)

    # Step 4: Create structured prompt
    prompt = f"""
You are a research assistant.

Answer the question using the detailed content of the provided context.
Do NOT just repeat the title.
Provide a clear, structured explanation based only on the context.
If the answer is not found in the context, say:
"The answer is not found in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    # Step 5: Call LLM
    answer = call_llm(prompt)

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks
    }
