# RAG-PaperBot 📚

RAG-PaperBot is a full-stack, AI-powered research assistant designed to ingest academic and research papers (PDFs) and provide precise, context-aware answers to user queries using a state-of-the-art **Retrieval-Augmented Generation (RAG)** pipeline.

It loads your PDFs, splits the text into meaningful chunks, embeds them using HuggingFace models, and stores them in a local FAISS vector database. When a question is asked, it queries the vector DB to retrieve verbatim context, which is then passed to an LLM (powered by Groq or OpenAI) to synthesize a comprehensive answer.

## Features ✨

- **Multi-PDF Support:** Upload one or multiple academic papers simultaneously.
- **Smart Document Processing:** Automatically extracts and chunks text using PyPDF and Recursive Character Splitting.
- **Semantic Search:** Uses `sentence-transformers/all-MiniLM-L6-v2` for high-quality dense vector embeddings.
- **Lightning Fast Inference:** Defaults to Groq (`llama3-8b-8192`) for near-instant responses (falls back to OpenAI GPT-3.5-Turbo).
- **Source Tracing:** Displays the exact document chunks and page numbers used to generate the answer.
- **Sleek UI:** Clean, responsive, dark-mode native interface built with Streamlit.

## Architecture

```text
       📄 PDF Upload 
            ↓ 
    ✂️ Text Processing  (PyPDFLoader & RecursiveCharacterTextSplitter)
            ↓
  🧠 Vector Embeddings  (HuggingFace all-MiniLM-L6-v2) 
            ↓ 
    💾 Vector Store     (FAISS CPU)
            ↓
   🔍 Semantic Search   (FAISS Retriever) 
            ↓
   🤖 LLM Generation    (Groq Llama-3 / OpenAI GPT-3.5)
            ↓
       💡 Output        (Answer + Retrieved Context)  
```

## Setup & Installation

**Prerequisites:** Python 3.9+

1. **Clone or Navigate to the Directory:**
   ```bash
   cd RAG-PaperBot
   ```

2. **Create a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # venv\Scripts\activate   # On Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and add API keys for the providers you wish to use (you only need one, but Groq is highly recommended for speed and cost):
   ```ini
   GROQ_API_KEY=your_groq_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Application

Execute the following command in the terminal to start the Streamlit frontend:

```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

## Usage Example

1. Open the app in your browser.
2. Under **Document Upload** in the sidebar, drag and drop test PDFs.
3. Click the **Process Documents** button and wait for the success message.
4. In the main panel, enter an academic query like:
   - *"What are the key findings of this paper?"*
   - *"Explain the transformer architecture mentioned."*
   - *"What are the main contributions of this research?"*
5. Click **Get Answer** and view the AI-generated synthesized response alongside the original source chunks.

## Tech Stack
- **Frontend Framework:** Streamlit
- **LLM Orchestration:** LangChain
- **Embeddings:** HuggingFace `sentence-transformers`
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Document Parsing:** PyPDF 
- **LLM Providers:** Groq (`llama3-8b`) / OpenAI (`gpt-3.5-turbo`)
