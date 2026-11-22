import tempfile
import streamlit as st
from rag import get_embedder, build_collection, answer

st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("📄 PDF Question-Answering")


@st.cache_resource
def load_embedder():
    return get_embedder()


embedder = load_embedder()

# --- Sidebar: upload and index ---
with st.sidebar:
    uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded is not None:
    # Only re-index when the file changes, not on every rerun
    if st.session_state.get("file_name") != uploaded.name:
        with st.spinner("Reading and indexing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            collection, n_chunks = build_collection(tmp_path, embedder)
        st.session_state.collection = collection
        st.session_state.file_name = uploaded.name
        st.session_state.messages = []  # new PDF -> new conversation
        st.sidebar.success(f"Indexed {n_chunks} chunks")
else:
    st.info("Upload a PDF in the sidebar to begin.")
    st.stop()

# --- Chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if question := st.chat_input("Ask about your PDF"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            text, chunks, path = answer(question, st.session_state.collection, embedder)
        st.markdown(text)
        st.caption(f"path: {path}")
        with st.expander("Sources"):
            for c in chunks:
                st.markdown(f"**Page {c['page']}** (similarity {c['similarity']:.2f})")
                st.caption(c["text"][:300])

    st.session_state.messages.append({"role": "assistant", "content": text})