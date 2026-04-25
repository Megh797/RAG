import streamlit as st
import os
import tempfile

# 🔥 Fix SSL issue (safe)
os.environ.pop("SSL_CERT_FILE", None)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline

st.set_page_config(page_title="Free RAG Chatbot", layout="wide")
st.title("📄 Chat with your PDFs (FREE RAG)")

# -------------------------
# SESSION STATE
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

DB_PATH = "faiss_db"

# -------------------------
# LOAD DOCUMENTS
# -------------------------
def load_docs(files):
    docs = []

    for file in files:
        # Use temp file (avoids overwrite bugs)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())

    return docs

# -------------------------
# SPLIT TEXT
# -------------------------
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

# -------------------------
# CREATE / LOAD VECTOR DB
# -------------------------
def get_vectorstore(texts=None):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load existing DB
    if os.path.exists(DB_PATH):
        return FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Create new DB
    if texts:
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_PATH)
        return db

    return None

# -------------------------
# LOAD LLM (FIXED)
# -------------------------
@st.cache_resource
def load_llm():
    pipe = pipeline(
        task="text-generation",  # ✅ FIXED
        model="google/flan-t5-base",
        max_length=512,
        temperature=0.2
    )
    return HuggingFacePipeline(pipeline=pipe)

# -------------------------
# UI
# -------------------------
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# Process PDFs
if uploaded_files:
    with st.spinner("Processing PDFs..."):
        docs = load_docs(uploaded_files)
        texts = split_docs(docs)

        vectorstore = get_vectorstore(texts)
        st.session_state.vectorstore = vectorstore

        st.success("✅ PDFs processed and stored!")

# Clear chat
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

# -------------------------
# CHAT SYSTEM
# -------------------------
if st.session_state.vectorstore is not None:

    llm = load_llm()

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        ),
        return_source_documents=True
    )

    user_input = st.chat_input("Ask your question...")

    if user_input:
        with st.spinner("Thinking... 🤔"):
            result = qa_chain({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })

            answer = result["answer"]
            sources = result["source_documents"]

            st.session_state.chat_history.append(
                (user_input, answer, sources)
            )

# -------------------------
# DISPLAY CHAT
# -------------------------
for item in st.session_state.chat_history:
    question, answer, sources = item

    st.chat_message("user").write(question)
    st.chat_message("assistant").write(answer)

    with st.expander("📚 Sources"):
        for i, doc in enumerate(sources):
            st.write(f"**Source {i+1}:**")
            st.write(doc.page_content[:300])
            st.write("---")