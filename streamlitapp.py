import streamlit as st
import openai
import dropbox
import os
import re
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Secrets ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
DROPBOX_TOKEN = st.secrets["DROPBOX_TOKEN"]
dbx = dropbox.Dropbox(DROPBOX_TOKEN)

st.title("Safety Document QA Assistant - FAISS Dropbox Version")

# --- Dropbox FAISS paths ---
FAISS_INDEXES = {
    "Industrial Hygiene FAISS Index": {
        "dropbox_folder": "/ih_faiss_index",
        "local_folder": "ih_faiss_index"
    }
}

def download_file_from_dropbox(dropbox_path, local_path):
    _, res = dbx.files_download(dropbox_path)
    with open(local_path, "wb") as f:
        f.write(res.content)

@st.cache_resource
def load_faiss_from_dropbox(index_name):
    info = FAISS_INDEXES[index_name]

    dropbox_folder = info["dropbox_folder"]
    local_folder = Path(info["local_folder"])
    local_folder.mkdir(exist_ok=True)

    # FAISS save_local creates these two files
    download_file_from_dropbox(
        f"{dropbox_folder}/index.faiss",
        local_folder / "index.faiss"
    )

    download_file_from_dropbox(
        f"{dropbox_folder}/index.pkl",
        local_folder / "index.pkl"
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = FAISS.load_local(
        str(local_folder),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore

def clean_and_render_response(text):
    text = re.sub(r'\\mug', r'\\mu g', text)
    text = re.sub(r'µg', r'\\mu g', text)
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph:
            st.markdown(paragraph, unsafe_allow_html=True)

selected_index = st.selectbox(
    "Select FAISS index:",
    list(FAISS_INDEXES.keys())
)

question = st.text_input("Ask your safety question:")

if "answer" not in st.session_state:
    st.session_state.answer = ""
    st.session_state.minimal_context = ""
    st.session_state.full_context = ""

if selected_index:
    try:
        vectorstore = load_faiss_from_dropbox(selected_index)
        st.success(f"Loaded FAISS index with {vectorstore.index.ntotal} vectors.")

    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        st.stop()

if st.button("Get Answer") and question:
    try:
        docs = vectorstore.similarity_search(question, k=4)

        minimal_context = ""
        full_context = ""

        for i, doc in enumerate(docs, start=1):
            metadata = doc.metadata

            source = metadata.get("source_file", metadata.get("source", "Unknown source"))
            title = metadata.get("document_title", "")
            page = metadata.get("page", None)
            chunk_number = metadata.get("chunk_number", None)

            section_info = f"Source {i}: {source}"

            if title:
                section_info += f" | Title: {title}"

            if page is not None:
                section_info += f" | Page: {page}"

            if chunk_number is not None:
                section_info += f" | Chunk: {chunk_number}"

            minimal_context += section_info + "\n"
            full_context += f"{section_info}\n{doc.page_content}\n\n"

        prompt = f"""Context:
{full_context}

Question:
{question}

Instructions:
1. Answer using only the provided context.
2. If the answer is not in the context, say that the provided documents do not contain enough information.
3. Cite the source name/page/chunk when useful.
4. Use clear safety-focused language.
5. Keep formatting simple.

Answer:"""

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful occupational safety and industrial hygiene assistant. Answer only from the retrieved context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=700
        )

        st.session_state.answer = response.choices[0].message.content
        st.session_state.minimal_context = minimal_context
        st.session_state.full_context = full_context

    except Exception as e:
        st.error(f"Error getting answer: {e}")

if st.session_state.answer:
    st.subheader("Answer:")
    clean_and_render_response(st.session_state.answer)

    st.subheader("Sources:")
    st.text(st.session_state.minimal_context)

    if st.button("Show Full Context"):
        st.subheader("Full Context:")
        st.text(st.session_state.full_context)
