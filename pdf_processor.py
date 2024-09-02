import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from vector_store import add_to_chroma
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data/pdf/new"
PROCESSED_PATH = "data/pdf/processed"

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks

def process_new_pdfs():
    documents = load_documents()
    if not documents:
        logger.info("No new documents to process.")
        return 0
    
    chunks = split_documents(documents)
    chunks_with_ids = calculate_chunk_ids(chunks)
    add_to_chroma(chunks_with_ids)
    
    processed_files = set()
    
    for doc in documents:
        source = doc.metadata.get("source")
        if source and source not in processed_files:
            try:
                destination = os.path.join(PROCESSED_PATH, os.path.basename(source))
                shutil.move(source, destination)
                processed_files.add(source)
                logger.info(f"Moved file: {source} to {destination}")
            except Exception as e:
                logger.error(f"Error processing file {source}: {str(e)}")
    return len(processed_files)
