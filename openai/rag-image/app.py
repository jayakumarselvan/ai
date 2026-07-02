"""
app.py
Simple Streamlit UI for:
  - Tab 1: RAG Q&A over uploaded text/PDF documents.
  - Tab 2: Image generation with OpenAI.

Run with:
    streamlit run app.py

Requires an OpenAI API key, either as the OPENAI_API_KEY env var or
entered in the sidebar at runtime.
"""

import io

import streamlit as st
from pypdf import PdfReader

from rag_engine import RAGStore, answer_question, get_client
from image_gen import generate_images

st.set_page_config(page_title="Simple RAG + Image Studio", page_icon="🔎", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar: API key + shared client
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
api_key_input = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Leave blank to use the OPENAI_API_KEY environment variable instead.",
)

client = None
client_error = None
try:
    client = get_client(api_key_input or None)
except ValueError as e:
    client_error = str(e)

if client_error:
    st.sidebar.warning(client_error)
else:
    st.sidebar.success("OpenAI client ready ✅")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "store" not in st.session_state:
    st.session_state.store = None  # created once client is available
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer, sources)

st.title("🔎 Simple RAG + 🎨 Image Studio")

tab_rag, tab_image = st.tabs(["📚 RAG Q&A", "🎨 Image Generation"])

# ---------------------------------------------------------------------------
# Tab 1: RAG
# ---------------------------------------------------------------------------
with tab_rag:
    st.subheader("1. Upload documents")
    uploaded_files = st.file_uploader(
        "Upload .txt, .md, or .pdf files to index",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    col_index, col_status = st.columns([1, 3])
    with col_index:
        index_clicked = st.button("Index documents", type="primary", disabled=client is None)

    if index_clicked and uploaded_files:
        if st.session_state.store is None:
            st.session_state.store = RAGStore(client=client)

        total_chunks = 0
        for f in uploaded_files:
            if f.type == "application/pdf" or f.name.lower().endswith(".pdf"):
                reader = PdfReader(io.BytesIO(f.read()))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                text = f.read().decode("utf-8", errors="ignore")

            n_chunks = st.session_state.store.add_document(text, source=f.name)
            total_chunks += n_chunks

        with col_status:
            st.success(f"Indexed {len(uploaded_files)} file(s) into {total_chunks} chunks.")

    if st.session_state.store and not st.session_state.store.is_empty():
        st.caption(f"📦 Store currently holds {len(st.session_state.store.chunks)} chunks.")

    st.divider()
    st.subheader("2. Ask a question")

    question = st.text_input("Your question", placeholder="What does the document say about...?")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=4)
    ask_clicked = st.button("Ask", disabled=client is None)

    if ask_clicked and question:
        if st.session_state.store is None or st.session_state.store.is_empty():
            st.warning("Please upload and index at least one document first.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                answer, retrieved = answer_question(
                    client, st.session_state.store, question, top_k=top_k
                )
            st.session_state.chat_history.insert(0, (question, answer, retrieved))

    for q, a, retrieved in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
            with st.expander(f"Sources ({len(retrieved)})"):
                for chunk, src, score in retrieved:
                    st.markdown(f"**{src}** — relevance `{score:.2f}`")
                    st.caption(chunk[:400] + ("..." if len(chunk) > 400 else ""))

# ---------------------------------------------------------------------------
# Tab 2: Image generation
# ---------------------------------------------------------------------------
with tab_image:
    st.subheader("Generate an image")
    img_prompt = st.text_area(
        "Prompt", placeholder="A watercolor painting of a cat reading a book"
    )
    col1, col2 = st.columns(2)
    with col1:
        n_images = st.slider("Number of images", 1, 4, 1)
    with col2:
        size = st.selectbox("Size", ["1024x1024", "1024x1536", "1536x1024"])

    generate_clicked = st.button("Generate", type="primary", disabled=client is None)

    if generate_clicked and img_prompt:
        with st.spinner("Generating image(s)..."):
            try:
                images = generate_images(client, img_prompt, n=n_images, size=size)
                cols = st.columns(len(images))
                for col, img_bytes in zip(cols, images):
                    col.image(img_bytes, use_container_width=True)
            except Exception as e:
                st.error(f"Image generation failed: {e}")

st.sidebar.divider()
st.sidebar.caption(
    "Models used: text-embedding-3-small (retrieval), gpt-4o-mini (answers), "
    "gpt-image-1 (images)."
)
