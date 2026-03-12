import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings

@st.cache_resource
def get_embeddings_model():
    """
    Instantiate and return the HuggingFace embeddings model.
    We use st.cache_resource to load this large model only once per session.
    """
    # all-MiniLM-L6-v2 is a good balance between performance and speed
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
