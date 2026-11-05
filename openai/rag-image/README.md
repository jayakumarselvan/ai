# Simple RAG + Image Studio

A minimal Streamlit app combining:
- **RAG Q&A**: upload `.txt` / `.md` / `.pdf` files, they're chunked and embedded
  (OpenAI `text-embedding-3-small`), and questions are answered with
  `gpt-4o-mini` using the most relevant chunks as context — no vector DB needed,
  just numpy cosine similarity.
- **Image generation**: text-to-image with OpenAI's `gpt-image-1`.

## Files
- `rag_engine.py` — chunking, embedding, in-memory vector store, retrieval + answer generation
- `image_gen.py` — image generation wrapper
- `app.py` — Streamlit UI
- `requirements.txt` — dependencies

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."   # or paste it into the sidebar at runtime
streamlit run app.py
```

## How it works

1. **Index**: Upload documents → text is split into ~800-character overlapping
   chunks → each chunk is embedded → stored in memory as a numpy matrix.
2. **Ask**: Your question is embedded → compared via cosine similarity against
   all chunks → top-k most relevant chunks are passed to the chat model as
   context → model answers grounded in that context.
3. **Generate images**: separate tab, straightforward prompt → `gpt-image-1` → image(s) shown inline.

## Notes / things you may want to extend
- Swap the in-memory store for FAISS/Chroma/pgvector if you need persistence or scale.
- Chunking is character-based for simplicity; token-based chunking (e.g. via `tiktoken`) would be more precise for large documents.
- No auth/rate-limiting — add before deploying publicly.
