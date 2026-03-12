import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Local imports
from embeddings import get_embeddings_model
from pdf_loader import load_and_split_pdfs
from vector_store import build_vector_store, get_retriever
from rag_pipeline import build_qa_chain, ask_question
from utils import save_uploaded_files, format_source_docs, cleanup_temp_dir

# Load Env Vars
load_dotenv()

# Check for required API Keys
has_groq = bool(os.getenv("GROQ_API_KEY"))
has_openai = bool(os.getenv("OPENAI_API_KEY"))

# ---- Page Config ----
st.set_page_config(
    page_title="RAG-PaperBot | AI Research Assistant", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Advanced CSS Styling ----
st.markdown("""
<style>
/* Import a modern font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* App background and main container */
.stApp {
    background-color: #0b0f19;
}

/* Custom Header with Gradient */
.hero-container {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    animation: fadeIn 0.8s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero-header {
    background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    padding-bottom: 0rem;
    letter-spacing: -1px;
}

.hero-subtitle {
    color: #9ca3af;
    font-size: 1.15rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Expander custom styling */
div[data-testid="stExpander"] details {
    border: 1px solid #1f2937;
    border-radius: 8px;
    background-color: #111827;
}

/* Status dots */
.status-dot-green {
    height: 12px;
    width: 12px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
    margin-right: 10px;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
}
.status-dot-red {
    height: 12px;
    width: 12px;
    background-color: #ef4444;
    border-radius: 50%;
    display: inline-block;
    margin-right: 10px;
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.6);
}

/* Sidebar styling tweaks */
[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #1f2937;
}

/* Striving for more modern buttons */
div.stButton > button:first-child {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    padding: 0.5rem 1rem;
    transition: all 0.3s ease;
}

div.stButton > button:first-child:hover {
    background-color: #1d4ed8;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    border-color: #1d4ed8;
    color: #ffffff;
}

/* Improve Chat Input */
[data-testid="stChatInput"] {
    border-radius: 12px;
    border: 1px solid #374151;
}

</style>
""", unsafe_allow_html=True)

# ---- Initialize Session State for Chat History ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Main Layout ----
st.markdown('<div class="hero-container">', unsafe_allow_html=True)
st.markdown('<h1 class="hero-header">RAG-PaperBot</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Upload <b>academic papers</b>, build a knowledge base, and ask intelligent questions instantly.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Warning if API key is missing
if not has_groq and not has_openai:
    st.error("⚠️ Setup Incomplete: Neither GROQ_API_KEY nor OPENAI_API_KEY is found in the `.env` file. The LLM cannot generate answers.", icon="🚨")

# ---- 1. Sidebar Logic & Configuration ----
with st.sidebar:
    st.markdown("### ⚙️ Configuration Setup")
    
    # Status Indicator
    if "qa_chain" in st.session_state:
        st.markdown('<div style="display: flex; align-items: center; padding: 12px; background-color: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; margin-bottom: 20px;"><span class="status-dot-green"></span><span style="font-weight: 600; color: #10b981;">System Ready</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="display: flex; align-items: center; padding: 12px; background-color: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; margin-bottom: 20px;"><span class="status-dot-red"></span><span style="font-weight: 600; color: #ef4444;">Awaiting Documents</span></div>', unsafe_allow_html=True)
        
    st.divider()
    
    st.markdown("### 📄 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF Files", 
        type="pdf", 
        accept_multiple_files=True,
        help="You can upload multiple research papers simultaneously."
    )
    
    process_btn = st.button("🔨 Process & Index Documents", use_container_width=True, type="primary")
    
    if process_btn:
        if not uploaded_files:
            st.error("Please select at least one PDF file.")
        else:
            with st.spinner("Analyzing and indexing documents..."):
                temp_dir = tempfile.mkdtemp()
                try:
                    # Pipeline execution
                    file_paths = save_uploaded_files(uploaded_files, temp_dir)
                    chunks = load_and_split_pdfs(file_paths)
                    
                    if not chunks:
                        st.error("Failed to extract text from PDFs.")
                        st.stop()
                        
                    embeddings = get_embeddings_model()
                    vector_store = build_vector_store(chunks, embeddings)
                    retriever = get_retriever(vector_store, k=5) # Retrieve top 5 chunks for broader context
                    qa_chain = build_qa_chain(retriever)
                    
                    # Store in session state
                    st.session_state["qa_chain"] = qa_chain
                    st.session_state.messages = [] # Clear history on new document upload
                    
                    st.success(f"Successfully processed {len(uploaded_files)} document(s) into {len(chunks)} contextual chunks.")
                except Exception as e:
                    st.error(f"Processing Failed: {str(e)}")
                finally:
                    cleanup_temp_dir(temp_dir)
                    
    st.divider()
    st.markdown("### 🛠️ Architecture Info")
    provider = "Groq" if has_groq else ("OpenAI" if has_openai else "None")
    model_name = "llama-3.1-8b-instant" if has_groq else ("gpt-3.5-turbo" if has_openai else "N/A")
    
    st.caption(f"**LLM Engine:** {provider} ({model_name})")
    st.caption("**Embedding Model:** HuggingFace `all-MiniLM-L6`")
    st.caption("**Vector Index:** FAISS CPU")

# ---- 2. Chat Interface Main Panel ----

# Render historical chat messages
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        # If it's an assistant message, optionally display the source chunks
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Source Documents (Context)"):
                st.markdown(message["sources"])

# Initial prompt if empty
if len(st.session_state.messages) == 0:
    st.info("👋 Welcome to RAG-PaperBot! Upload a PDF paper in the sidebar and click **Process & Index Documents** to begin.")

# Chat Input Handler
if prompt := st.chat_input("Ask a question about your documents..."):
    
    # Immediately render the user's question
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    
    # Store user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if "qa_chain" not in st.session_state:
        # Error handling if pipeline not ready
        error_msg = "I haven't ingested any documents yet. Please use the sidebar to upload and process a PDF first."
        with st.chat_message("assistant", avatar="🤖"):
            st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        # Process the QA request
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Scanning knowledge base for answers..."):
                try:
                    qa_chain = st.session_state["qa_chain"]
                    answer, source_docs = ask_question(qa_chain, prompt)
                    
                    st.markdown(answer)
                    
                    # Formatting sources
                    formatted_sources = format_source_docs(source_docs)
                    with st.expander("📚 View Source Documents (Context)"):
                        st.markdown(formatted_sources)
                        
                    # Save Assistant response to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": formatted_sources
                    })
                    
                except Exception as e:
                    st.error(f"Error connecting to LLM or interpreting context: {str(e)}")
