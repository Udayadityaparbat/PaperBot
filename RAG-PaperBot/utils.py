import os
import shutil
import tempfile

def save_uploaded_files(uploaded_files, temp_dir: str) -> list[str]:
    """
    Accepts Streamlit UploadedFile objects and writes them out to a temporary directory.
    Returns the list of absolute file paths meant for document parsing.
    """
    file_paths = []
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    for uploaded_file in uploaded_files:
        # Create a path saving the original filename safely
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_paths.append(temp_path)
        
    return file_paths

def format_source_docs(source_documents) -> str:
    """
    Format the list of derived chunks back into a readable Streamlit markdown string.
    """
    formatted_str = ""
    for i, doc in enumerate(source_documents):
        source = doc.metadata.get('source', 'Unknown Document')
        page = doc.metadata.get('page', 'Unknown Page')
        
        formatted_str += f"**Source {i+1}**: `{source}` (Page {page})\n\n"
        formatted_str += f"> {doc.page_content.replace(chr(10), ' ')}\n\n"
        formatted_str += "---\n\n"
        
    return formatted_str

def cleanup_temp_dir(temp_dir: str):
    """
    Removes the temporary directory completely.
    """
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
