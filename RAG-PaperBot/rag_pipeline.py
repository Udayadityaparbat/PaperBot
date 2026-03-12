import os
from dotenv import load_dotenv

# LLM Providers
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Chains & Prompts
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load env variables globally
load_dotenv()

def get_llm():
    """
    Initialize the LLM. 
    Attempts to use Groq if the API key is set, otherwise falls back to OpenAI.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if groq_api_key:
        # Llama-3-8b via Groq is extremely fast and a great default
        return ChatGroq(
            api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.0
        )
    elif openai_api_key:
        # Fallback to standard OpenAI GPT-3.5-Turbo
        return ChatOpenAI(
            api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.0
        )
    else:
        raise ValueError("Neither GROQ_API_KEY nor OPENAI_API_KEY could be found in the environment variables.")

def build_qa_chain(retriever):
    """
    Build a RAG chain to answer user questions using the retrieved context.
    Returns a unified chain object using RetrievalQA.
    """
    llm = get_llm()
    
    template = """You are a helpful and intelligent AI research assistant built to answer questions about academic research papers.
Use the following pieces of retrieved context to answer the user's question. 
If you don't know the answer or if the context doesn't contain the information, just say that you don't know, don't try to make up an answer.
Always try to use an academic, professional tone.

Context: {context}

Question: {question}

Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Building a standard RetrievalQA chain
    # It takes the question, fetches context from the retriever, and passes both to the LLM.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return qa_chain

def ask_question(qa_chain, query: str):
    """
    Send the user query to the QA chain.
    Returns a tuple of (Answer String, Source Documents List).
    """
    result = qa_chain.invoke({"query": query})
    return result["result"], result["source_documents"]
