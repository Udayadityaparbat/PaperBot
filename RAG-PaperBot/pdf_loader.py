import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_and_split_pdfs(file_paths: list[str]) -> list[Document]:
    """
    Load a list of PDF file paths, parse them, and split them into chunks.
    """
    all_chunks = []
    
    # We use a relatively standard chunk size for research papers (e.g., preserving paragraphs)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Add metadata about the source file (just the filename, not full temp path)
            file_name = os.path.basename(file_path)
            for doc in docs:
                if 'source' in doc.metadata:
                    doc.metadata['source'] = file_name
                    
            # Split the document into chunks
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return all_chunks
