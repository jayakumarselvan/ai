"""
rag_engine.py
A minimal, dependency-light Retrieval-Augmented Generation (RAG) engine
built on top of the OpenAI API.

Pipeline:
  1. Chunk raw text into overlapping windows.
  2. Embed each chunk with an OpenAI embedding model.
  3. Store embeddings in-memory as a numpy matrix (no external vector DB).
  4. At query time, embed the question, rank chunks by cosine similarity,
     and feed the top-k chunks to a chat model as context.

This is intentionally simple (no FAISS/Chroma/Pinecone) so it's easy to
read, extend, and swap out pieces later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def get_client(api_key: str | None = None) -> OpenAI:
    """Create an OpenAI client, preferring an explicitly passed key."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "No OpenAI API key found. Set OPENAI_API_KEY or pass one in."
        )
    return OpenAI(api_key=key)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks (by character count).

    chunk_size / overlap are in characters, which is a simple and
    model-agnostic way to keep chunks small enough for embedding and
    context windows without needing a tokenizer.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap  # step forward, keeping some overlap
    return chunks


@dataclass
class RAGStore:
    """In-memory vector store: parallel lists of chunks + their embeddings."""

    client: OpenAI
    chunks: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)  # e.g. filename per chunk
    embeddings: np.ndarray | None = None  # shape (n_chunks, dim)

    def add_document(self, text: str, source: str = "document") -> int:
        """Chunk + embed a document, appending it to the store.

        Returns the number of chunks added.
        """
        new_chunks = chunk_text(text)
        if not new_chunks:
            return 0

        new_embeddings = self._embed(new_chunks)

        self.chunks.extend(new_chunks)
        self.sources.extend([source] * len(new_chunks))

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        return len(new_chunks)

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Call the OpenAI embeddings API in one batch and L2-normalize."""
        resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # Normalize so dot product == cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return vectors / norms

    def search(self, query: str, top_k: int = 4) -> List[Tuple[str, str, float]]:
        """Return the top_k (chunk, source, score) tuples for a query."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_vec = self._embed([query])[0]  # already normalized
        scores = self.embeddings @ query_vec  # cosine similarity, shape (n,)

        top_k = min(top_k, len(self.chunks))
        top_idx = np.argsort(-scores)[:top_k]

        return [(self.chunks[i], self.sources[i], float(scores[i])) for i in top_idx]

    def is_empty(self) -> bool:
        return self.embeddings is None or len(self.chunks) == 0


def answer_question(
    client: OpenAI,
    store: RAGStore,
    question: str,
    top_k: int = 4,
    model: str = CHAT_MODEL,
) -> Tuple[str, List[Tuple[str, str, float]]]:
    """Retrieve relevant chunks and ask the chat model to answer using them.

    Returns (answer_text, retrieved_chunks).
    """
    retrieved = store.search(question, top_k=top_k)

    if not retrieved:
        context_block = "No documents have been indexed yet."
    else:
        context_block = "\n\n".join(
            f"[Source: {src} | relevance {score:.2f}]\n{chunk}"
            for chunk, src, score in retrieved
        )

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the "
        "provided context. If the context doesn't contain the answer, say "
        "you don't have enough information rather than making something up. "
        "Cite which source(s) you used when relevant."
    )

    user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    return answer, retrieved
