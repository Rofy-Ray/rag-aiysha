import os
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from vector_store import add_to_chroma
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data/pdf/new"
PROCESSED_PATH = "data/pdf/processed"
VECTORIZED_FILE = os.path.join(PROCESSED_PATH, "vectorized.txt")

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
    
    current_count = 0
    processed_pdfs = []
    if os.path.exists(VECTORIZED_FILE):
        with open(VECTORIZED_FILE, 'r') as f:
            lines = f.readlines()
            if lines:
                current_count = int(lines[0].strip())
                processed_pdfs = [line.strip() for line in lines[1:]]
    
    for doc in documents:
        source = doc.metadata.get("source")
        if source and source not in processed_files:
            try:
                os.remove(source)
                processed_files.add(source)
                processed_pdfs.append(os.path.basename(source))
                logger.info(f"Processed and deleted file: {source}")
            except Exception as e:
                logger.error(f"Error processing file {source}: {str(e)}")
    new_count = current_count + len(processed_files)
    with open(VECTORIZED_FILE, 'w') as f:
        f.write(f"{new_count}\n")
        for pdf in processed_pdfs:
            f.write(f"{pdf}\n")
    logger.info(f"Updated vectorized.txt with {len(processed_files)} new files. Total count: {new_count}")
    return len(processed_files)
