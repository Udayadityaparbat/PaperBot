import os
from langchain_community.vectorstores import FAISS

def build_vector_store(chunks, embeddings_model):
    """
    Given a list of document chunks and an embedding model,
    build and return a FAISS vector store.
    """
    if not chunks:
        raise ValueError("No chunks provided to build the vector store.")
    
    # from_documents takes chunks, encodes them using the embeddings_model, 
    # and populates an in-memory FAISS index.
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    return vector_store

def save_vector_store(vector_store, persist_directory: str):
    """
    Save the FAISS vector store to the given directory path.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        
    vector_store.save_local(persist_directory)

def load_vector_store(persist_directory: str, embeddings_model) -> FAISS:
    """
    Load an existing FAISS vector store from disk.
    """
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Directory {persist_directory} does not exist.")
        
    # We load with allow_dangerous_deserialization=True because FAISS saves picking files.
    # It shouldn't be loaded from untrusted sources, but we generate it ourselves here.
    return FAISS.load_local(
        persist_directory, 
        embeddings_model, 
        allow_dangerous_deserialization=True
    )

def get_retriever(vector_store: FAISS, k: int = 4):
    """
    Return a base Retriever from the vector store interface, to be used in QA chains.
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
