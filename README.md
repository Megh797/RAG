# 📄 RAG PDF Chatbot (Offline & Free)

A production-ready **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDFs and interact with them through a conversational AI interface — completely offline and without any paid APIs.

---

## 🚀 Demo

👉 Upload PDFs → Ask questions → Get context-aware answers with sources

---

## ✨ Features

* 📂 Multi-PDF upload support
* 🔍 Semantic search using FAISS
* 🧠 Context-aware answers using RAG
* 💬 Conversational memory (chat history)
* 📚 Source citations for transparency
* ⚡ Persistent vector database (no recomputation)
* 🆓 Fully offline (no OpenAI or paid APIs)

---

## 🧠 Tech Stack

* **Frontend:** Streamlit
* **Backend:** LangChain
* **Vector Store:** FAISS
* **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
* **LLM:** FLAN-T5 (Hugging Face)
* **Language:** Python

---

## 📁 Project Structure

```
rag-pdf-chatbot/
│── app.py
│── requirements.txt
│── README.md
│── faiss_db/        # Stored vector database
│── temp.pdf         # Temporary uploaded file
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

---

### 2️⃣ Create virtual environment 

```bash
conda create -n rag_project python=3.10
conda activate rag_project
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📌 How It Works

1. Upload PDF documents
2. Text is split into chunks
3. Embeddings are generated using Sentence Transformers
4. Stored in FAISS vector database
5. User query → similarity search → relevant chunks
6. LLM generates answer based on retrieved context

---

## 📸 Example Workflow

* Upload research paper 📄
* Ask: *"What is the main conclusion?"*
* Get:

  * ✅ Accurate answer
  * 📚 Source references

---

## ⚠️ Limitations

* Responses depend on document quality
* Open-source models are less powerful than GPT-4
* Initial model download may take time

---

## 🚀 Future Improvements

* 🔹 Highlight answers inside PDFs
* 🔹 Add document preview
* 🔹 Improve UI (dark mode, sidebar)
* 🔹 Deploy on cloud (Streamlit / Render)
* 🔹 Add hybrid search (BM25 + vector)

---

## 💼 Resume Description

**RAG-based PDF Chatbot (Offline)**

* Built a Retrieval-Augmented Generation system using LangChain and FAISS
* Implemented semantic search with sentence-transformer embeddings
* Designed conversational AI with memory and source attribution
* Developed interactive UI using Streamlit

---




