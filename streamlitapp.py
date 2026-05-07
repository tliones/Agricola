import streamlit as st
import dropbox
import re
from pathlib import Path
from openai import OpenAI

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# =========================
# Streamlit App Setup
# =========================

st.set_page_config(
    page_title="Safety Document QA Assistant",
    layout="wide"
)

st.title("Safety Document QA Assistant - FAISS Dropbox Version")


# =========================
# Secrets Check
# =========================

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

if "DROPBOX_TOKEN" not in st.secrets:
    st.error("Missing DROPBOX_TOKEN in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
dbx = dropbox.Dropbox(st.secrets["DROPBOX_TOKEN"])


# =========================
# Dropbox FAISS Index Paths
# =========================

FAISS_INDEXES = {
    "Industrial Hygiene FAISS Index": {
        "dropbox_folder": "/ih_faiss_index",
        "local_folder": "ih_faiss_index"
    }
}


# =========================
# Helper Functions
# =========================

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

    download_file_from_dropbox(
        f"{dropbox_folder}/index.faiss",
        local_folder / "index.faiss"
    )

    download_file_from_dropbox(
        f"{dropbox_folder}/index.pkl",
        local_folder / "index.pkl"
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=st.secrets["OPENAI_API_KEY"]
    )

    vectorstore = FAISS.load_local(
        str(local_folder),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def get_metadata_options(vectorstore, field):
    values = set()

    for doc in vectorstore.docstore._dict.values():
        value = doc.metadata.get(field)

        if isinstance(value, list):
            for item in value:
                if item:
                    values.add(str(item))
        elif value:
            values.add(str(value))

    return sorted(values)


def metadata_matches(doc, selected_docs, selected_orgs, selected_jurisdictions, selected_types):
    metadata = doc.metadata

    source_file = str(metadata.get("source_file", "Unknown"))
    organization = str(metadata.get("organization", "Unknown"))
    jurisdiction = str(metadata.get("jurisdiction", "Unknown"))
    document_type = str(metadata.get("document_type", "Unknown"))

    if selected_docs and source_file not in selected_docs:
        return False

    if selected_orgs and organization not in selected_orgs:
        return False

    if selected_jurisdictions and jurisdiction not in selected_jurisdictions:
        return False

    if selected_types and document_type not in selected_types:
        return False

    return True


def clean_and_render_response(text):
    text = re.sub(r'\\mug', r'\\mu g', text)
    text = re.sub(r'µg', r'\\mu g', text)
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph:
            st.markdown(paragraph, unsafe_allow_html=True)


def build_context(docs):
    minimal_context = ""
    full_context = ""

    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata

        source = metadata.get("source_file", metadata.get("source", "Unknown source"))
        title = metadata.get("document_title", "")
        organization = metadata.get("organization", "")
        jurisdiction = metadata.get("jurisdiction", "")
        doc_type = metadata.get("document_type", "")
        page = metadata.get("page", None)
        chunk_number = metadata.get("chunk_number", None)

        section_info = f"Source {i}: {source}"

        if title:
            section_info += f" | Title: {title}"

        if organization:
            section_info += f" | Organization: {organization}"

        if jurisdiction:
            section_info += f" | Jurisdiction: {jurisdiction}"

        if doc_type:
            section_info += f" | Type: {doc_type}"

        if page is not None:
            section_info += f" | Page: {page}"

        if chunk_number is not None:
            section_info += f" | Chunk: {chunk_number}"

        minimal_context += section_info + "\n"
        full_context += f"{section_info}\n\n{doc.page_content}\n\n---\n\n"

    return minimal_context, full_context


# =========================
# Load FAISS Index
# =========================

selected_index = st.selectbox(
    "Select FAISS index:",
    list(FAISS_INDEXES.keys())
)

try:
    vectorstore = load_faiss_from_dropbox(selected_index)
    st.success(f"Loaded FAISS index with {vectorstore.index.ntotal} vectors.")
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.stop()


# =========================
# Metadata Filters
# =========================

st.sidebar.header("Search Filters")

available_docs = get_metadata_options(vectorstore, "source_file")
available_orgs = get_metadata_options(vectorstore, "organization")
available_jurisdictions = get_metadata_options(vectorstore, "jurisdiction")
available_types = get_metadata_options(vectorstore, "document_type")

selected_docs = st.sidebar.multiselect(
    "Documents",
    available_docs,
    default=available_docs
)

selected_orgs = st.sidebar.multiselect(
    "Organizations",
    available_orgs,
    default=[]
)

selected_jurisdictions = st.sidebar.multiselect(
    "Jurisdictions",
    available_jurisdictions,
    default=[]
)

selected_types = st.sidebar.multiselect(
    "Document Types",
    available_types,
    default=[]
)

search_k = st.sidebar.slider(
    "Initial retrieval depth",
    min_value=5,
    max_value=50,
    value=20,
    step=5
)

final_k = st.sidebar.slider(
    "Final chunks sent to LLM",
    min_value=2,
    max_value=10,
    value=4,
    step=1
)


# =========================
# Main Question UI
# =========================

question = st.text_input("Ask your safety question:")

if "answer" not in st.session_state:
    st.session_state.answer = ""
    st.session_state.minimal_context = ""
    st.session_state.full_context = ""
    st.session_state.filtered_docs_count = 0


if st.button("Get Answer") and question:
    try:
        retrieved_docs = vectorstore.similarity_search(
            question,
            k=search_k
        )

        filtered_docs = [
            doc for doc in retrieved_docs
            if metadata_matches(
                doc,
                selected_docs,
                selected_orgs,
                selected_jurisdictions,
                selected_types
            )
        ]

        final_docs = filtered_docs[:final_k]

        st.session_state.filtered_docs_count = len(filtered_docs)

        if not final_docs:
            st.warning("No matching chunks found with the selected filters. Try broadening your filters.")
            st.stop()

        minimal_context, full_context = build_context(final_docs)

        prompt = f"""Context:
{full_context}

Question:
{question}

Instructions:
1. Answer using only the provided context.
2. If the answer is not in the context, say the provided documents do not contain enough information.
3. Cite the source name, page, and chunk when useful.
4. Use clear occupational safety and industrial hygiene language.
5. Keep formatting simple and practical.

Answer:"""

        response = client.chat.completions.create(
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
            max_tokens=900
        )

        st.session_state.answer = response.choices[0].message.content
        st.session_state.minimal_context = minimal_context
        st.session_state.full_context = full_context

    except Exception as e:
        st.error(f"Error getting answer: {e}")


# =========================
# Output
# =========================

if st.session_state.answer:
    st.subheader("Answer:")
    clean_and_render_response(st.session_state.answer)

    st.subheader("Sources Used:")
    st.text(st.session_state.minimal_context)

    st.caption(f"Filtered matching chunks found: {st.session_state.filtered_docs_count}")

    with st.expander("Show Full Retrieved Context"):
        st.text(st.session_state.full_context)
