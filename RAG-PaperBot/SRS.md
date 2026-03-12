# Software Requirements Specification (SRS)
## RAG-PaperBot — AI-Powered Research Assistant

**Document Version:** 1.0  
**Date:** March 12, 2026  
**Status:** Draft  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [Functional Requirements](#3-functional-requirements)
4. [Non-Functional Requirements](#4-non-functional-requirements)
5. [System Architecture](#5-system-architecture)
6. [External Interface Requirements](#6-external-interface-requirements)
7. [Data Requirements](#7-data-requirements)
8. [Constraints & Assumptions](#8-constraints--assumptions)

---

## 1. Introduction

### 1.1 Purpose
This Software Requirements Specification (SRS) document describes the functional and non-functional requirements for **RAG-PaperBot**, a full-stack, AI-powered research assistant application built using the Retrieval-Augmented Generation (RAG) paradigm. The document is intended for developers, maintainers, and stakeholders seeking to understand the design and expected behavior of the system.

### 1.2 Scope
RAG-PaperBot enables users to upload academic and research papers in PDF format, processes them into a searchable vector knowledge base, and generates precise, context-aware natural language answers to user queries using a Large Language Model (LLM). The application operates entirely locally (except for LLM API calls) on a single-user basis through a browser-based Streamlit interface.

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|---|---|
| RAG | Retrieval-Augmented Generation |
| LLM | Large Language Model |
| FAISS | Facebook AI Similarity Search |
| PDF | Portable Document Format |
| NLP | Natural Language Processing |
| API | Application Programming Interface |
| SRS | Software Requirements Specification |
| UI | User Interface |

### 1.4 References
- [LangChain Documentation](https://docs.langchain.com)
- [Groq API Documentation](https://console.groq.com/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [FAISS Documentation](https://faiss.ai/)
- [HuggingFace sentence-transformers Documentation](https://sbert.net/)

---

## 2. Overall Description

### 2.1 Product Perspective
RAG-PaperBot is a standalone Python-based web application. It integrates third-party LLM inference APIs (Groq, OpenAI) with locally run embedding and vector-search components. It is not part of a larger software ecosystem and requires no external database server.

### 2.2 Product Features Summary
- Multi-PDF ingestion and text extraction
- Automatic text chunking with configurable overlap
- Semantic vector embedding using HuggingFace models
- In-memory FAISS vector store with similarity search
- LLM-generated answers via Groq (primary) or OpenAI (fallback)
- Chat-based UI with persistent session history
- Source document citation alongside each generated answer
- API key-based LLM provider selection

### 2.3 User Classes and Characteristics

| User Class | Description |
|---|---|
| **Research Student** | Uploads research papers and queries specific findings, terminology, or methodologies |
| **Academic Professional** | Needs rapid extraction and cross-paper synthesis of domain knowledge |
| **Developer / Tester** | Configures the system, manages API keys and environment, and tests the RAG pipeline |

### 2.4 Operating Environment
- **OS:** macOS, Linux, Windows (Python 3.9+)
- **Runtime:** Python 3.9 or higher
- **Interface:** Web browser (Chrome, Firefox, Safari, Edge)
- **Hardware:** Modern CPU; GPU not required
- **Network:** Internet connection required for LLM inference (Groq/OpenAI API calls)

---

## 3. Functional Requirements

### 3.1 Document Ingestion Module (`pdf_loader.py`)

**FR-01:** The system **shall** accept one or more PDF files as input through the UI.

**FR-02:** The system **shall** extract all readable text content from every page of each uploaded PDF using `PyPDFLoader`.

**FR-03:** The system **shall** split each loaded document into text chunks using `RecursiveCharacterTextSplitter` with the following parameters:
  - `chunk_size`: 1000 characters
  - `chunk_overlap`: 200 characters
  - Separator hierarchy: `["\n\n", "\n", " ", ""]`

**FR-04:** Each chunk **shall** retain metadata including the source filename.

**FR-05:** The system **shall** handle missing or corrupt PDF files gracefully, logging a warning and continuing without crashing.

---

### 3.2 Embedding Module (`embeddings.py`)

**FR-06:** The system **shall** compute dense vector embeddings for all document chunks using the HuggingFace model `sentence-transformers/all-MiniLM-L6-v2`.

**FR-07:** All embeddings **shall** be L2-normalized to ensure cosine similarity equivalence in dot-product search.

**FR-08:** The embedding model **shall** be loaded once per session and cached using Streamlit's `@st.cache_resource` decorator to prevent repeated and expensive model reloads.

---

### 3.3 Vector Store Module (`vector_store.py`)

**FR-09:** The system **shall** construct an in-memory FAISS index from the computed chunk embeddings upon each document processing action.

**FR-10:** The system **shall** support saving the FAISS index to a local directory on disk.

**FR-11:** The system **shall** support loading a previously persisted FAISS index from disk.

**FR-12:** The system **shall** expose a LangChain-compatible `Retriever` interface configured with similarity search, returning the top `k=5` most relevant chunks for each user query.

---

### 3.4 RAG Pipeline Module (`rag_pipeline.py`)

**FR-13:** The system **shall** attempt to initialize the LLM using the `GROQ_API_KEY` (Groq's `llama-3.1-8b-instant` model) as the primary provider.

**FR-14:** If the `GROQ_API_KEY` is not present, the system **shall** fall back to OpenAI's `gpt-3.5-turbo` model using the `OPENAI_API_KEY`.

**FR-15:** If neither key is found, the system **shall** raise a `ValueError` with a descriptive error message.

**FR-16:** The system **shall** construct a `RetrievalQA` chain using LangChain's `"stuff"` chain type, combining retrieved context and the user's question into a single prompt.

**FR-17:** The prompt template **shall** instruct the LLM to act as an academic research assistant, use only the provided retrieved context, and respond with "I don't know" when context is insufficient.

**FR-18:** The QA chain **shall** return both the generated answer string and the list of source `Document` objects used.

---

### 3.5 User Interface Module (`app.py`)

**FR-19:** The system **shall** display a dark-mode, browser-based Streamlit user interface.

**FR-20:** The system **shall** present a sidebar containing:
  - A live system status indicator (Ready / Awaiting Documents)
  - A file uploader accepting multiple `.pdf` files
  - A "Process & Index Documents" action button
  - Architecture information panel (LLM engine, embedding model, vector index)

**FR-21:** The system **shall** display a warning banner at the top of the main panel if no valid API key is configured in the environment.

**FR-22:** The system **shall** render a scrollable, persistent chat history on the main panel with distinct user (🧑‍💻) and assistant (🤖) avatars.

**FR-23:** The system **shall** display the assistant's generated answer in real-time alongside an expandable "View Source Documents" section.

**FR-24:** The chat history **shall** be stored in Streamlit's `session_state` and persist for the lifetime of the user's browser session.

**FR-25:** The chat history **shall** be cleared whenever new documents are processed and indexed.

**FR-26:** If the user queries without first uploading documents, the system **shall** return a friendly error message and not invoke the QA pipeline.

---

## 4. Non-Functional Requirements

### 4.1 Performance

**NFR-01:** Initial embedding model load time should be under **30 seconds** on standard consumer hardware (the model is cached after the first load).

**NFR-02:** FAISS index construction for a typical single research paper (< 50 pages) should complete in under **15 seconds**.

**NFR-03:** LLM response latency should be under **5 seconds** for typical queries when using Groq's inference endpoint (subject to network conditions).

### 4.2 Usability

**NFR-04:** The UI must be operable without any command-line interaction once the application is running.

**NFR-05:** All error states (missing API keys, failed document parsing, LLM connection errors) must be communicated to the user via descriptive on-screen messages — never as raw Python tracebacks.

### 4.3 Reliability

**NFR-06:** The application must not crash when a user uploads a PDF that contains no extractable text (e.g., a scanned image-only PDF); it must display a suitable error message.

**NFR-07:** The application must not crash if the LLM API returns an unexpected error; instead, it must catch the exception and display an error to the user.

### 4.4 Security

**NFR-08:** API keys must be stored in a `.env` file and loaded via `python-dotenv`. They must **never** be hardcoded in source code or committed to version control.

**NFR-09:** The `.env` file must be listed in `.gitignore`.

**NFR-10:** FAISS indexes loaded from disk should only be loaded from trusted, self-generated sources, as deserialization is inherently trusted (`allow_dangerous_deserialization=True`).

### 4.5 Maintainability

**NFR-11:** The application code must follow a modular architecture with clearly separated concerns: ingestion, embedding, vector storage, RAG pipeline, and UI layers.

**NFR-12:** Each module must contain docstrings explaining its responsibilities.

### 4.6 Portability

**NFR-13:** The application must run on macOS, Linux, and Windows environments with Python 3.9+.

**NFR-14:** All dependencies must be pinned or specified in `requirements.txt` to allow deterministic environment reconstruction.

---

## 5. System Architecture

### 5.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         User Browser                        │
│                    (Streamlit Web Interface)                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │          app.py            │
              │  (Orchestration / UI Layer) │
              └──┬─────────┬──────────┬────┘
                 │         │          │
    ┌────────────▼──┐  ┌───▼──────┐  ┌▼──────────────┐
    │  pdf_loader   │  │embeddings│  │ rag_pipeline   │
    │  .py          │  │.py       │  │ .py            │
    │(Chunking)     │  │(HFEmbeds)│  │(QA Chain/LLM)  │
    └─────────┬─────┘  └────┬─────┘  └───────┬────────┘
              │             │                 │
              └──────────┬──┘         ┌───────▼──────────────┐
                         │            │ External APIs:        │
               ┌─────────▼──────┐     │  - Groq Inference    │
               │  vector_store  │     │  - OpenAI Inference  │
               │  .py (FAISS)   │     └──────────────────────┘
               └────────────────┘
```

### 5.2 RAG Pipeline Data Flow

```
📄 PDF Upload
     ↓
✂️ Text Extraction & Chunking   (PyPDFLoader + RecursiveCharacterTextSplitter)
     ↓
🧠 Vector Embeddings            (HuggingFace all-MiniLM-L6-v2)
     ↓
💾 Vector Store                 (FAISS In-Memory Index)
     ↓
🔍 Semantic Retrieval           (Top-K Similarity Search, k=5)
     ↓
📝 Prompt Construction          (Context + User Question + System Prompt)
     ↓
🤖 LLM Generation               (Groq llama-3.1-8b-instant / OpenAI gpt-3.5-turbo)
     ↓
💡 Output                       (Answer + Source Document Citations)
```

---

## 6. External Interface Requirements

### 6.1 User Interfaces
- A Streamlit-based single-page web application accessible at `http://localhost:8501`.
- Two-panel layout: a left sidebar for configuration and a main panel for chat interaction.
- Responsive dark mode design using custom CSS injected via `st.markdown`.

### 6.2 API Interfaces

| Provider | Endpoint Used | Authentication | Model Used |
|---|---|---|---|
| Groq | `api.groq.com` | `GROQ_API_KEY` (env var) | `llama-3.1-8b-instant` |
| OpenAI | `api.openai.com` | `OPENAI_API_KEY` (env var) | `gpt-3.5-turbo` |
| HuggingFace | Downloaded model | None (local inference) | `all-MiniLM-L6-v2` |

### 6.3 Software Interfaces

| Library | Version Constraint | Purpose |
|---|---|---|
| `streamlit` | Latest stable | Web UI framework |
| `langchain` | Latest stable | LLM orchestration & chains |
| `langchain-community` | Latest stable | PDF loaders, FAISS, HuggingFace embeddings |
| `langchain-groq` | Latest stable | Groq LLM integration |
| `langchain-openai` | Latest stable | OpenAI LLM integration |
| `langchain-core` | Latest stable | Core prompts and interfaces |
| `faiss-cpu` | Latest stable | Vector similarity search |
| `sentence-transformers` | Latest stable | Text embedding models |
| `pypdf` | Latest stable | PDF text extraction |
| `python-dotenv` | Latest stable | `.env` file loading |
| `tiktoken` | Latest stable | Token counting for OpenAI models |

---

## 7. Data Requirements

### 7.1 Input Data
- **PDF Files:** One or more valid PDF documents. Text must be digitally encoded (not scanned images) for reliable extraction.

### 7.2 Intermediate Data
- **Document Chunks:** Lists of LangChain `Document` objects, each containing `page_content` (string) and `metadata` (dict with `source`, `page` keys).
- **Embeddings Vector:** 384-dimensional float vectors (output of `all-MiniLM-L6-v2`).
- **FAISS Index:** In-memory `IndexFlatL2` structure for fast nearest-neighbor lookup.

### 7.3 Output Data
- **Answer:** A natural language string generated by the LLM.
- **Source Documents:** A list of the top-K retrieved `Document` chunks, displayed with their source filename and page number.

### 7.4 Session State
All runtime data (processed QA chain, chat history) is stored in Streamlit's `st.session_state` dictionary and is **ephemeral** — it does not persist across browser refresh or app restarts.

---

## 8. Constraints & Assumptions

### 8.1 Constraints
- **C-01:** The application requires at least one valid LLM API key (`GROQ_API_KEY` or `OPENAI_API_KEY`) to generate answers.
- **C-02:** The embedding model (`all-MiniLM-L6-v2`) requires an internet connection or local cache on first use.
- **C-03:** The FAISS vector store is in-memory only; all indexed data is lost upon application restart unless explicitly saved to disk using `save_vector_store()`.
- **C-04:** Image-based or scanned PDFs will produce no extractable text; the system will generate 0 chunks and fail gracefully.
- **C-05:** LLM output quality is bounded by the quality and completeness of the ingested context chunks.

### 8.2 Assumptions
- **A-01:** Users have Python 3.9+ and `pip` installed.
- **A-02:** Users have access to at least one valid LLM provider API key (Groq recommended).
- **A-03:** Uploaded PDF files are legitimate, text-based academic documents.
- **A-04:** The application operates in a single-user, local deployment context and does not require authentication or multi-tenancy.

---

*End of Document*
