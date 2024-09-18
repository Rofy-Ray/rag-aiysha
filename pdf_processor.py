import os
import logging
import tempfile
from google.cloud import storage
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from vector_store import add_to_chroma


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUCKET_NAME = "aiysha-convos"
DATA_PATH = "pdf/new/"
VECTORIZED_FILE = "pdf/processed/vectorized.txt"

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def load_documents():
    documents = []
    blobs = bucket.list_blobs(prefix=DATA_PATH)
    for blob in blobs:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            blob.download_to_filename(temp_file.name)
            loader = PyPDFLoader(temp_file.name)
            documents.extend(loader.load())
            os.unlink(temp_file.name)
    return documents

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
    vectorized_blob = bucket.blob(VECTORIZED_FILE)
    if vectorized_blob.exists():
        with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as temp_file:
            vectorized_blob.download_to_filename(temp_file.name)
            with open(temp_file.name, 'r') as f:
                lines = f.readlines()
                if lines:
                    current_count = int(lines[0].strip())
                    processed_pdfs = [line.strip() for line in lines[1:]]
            os.unlink(temp_file.name)
    
    for doc in documents:
        source = doc.metadata.get("source")
        if source and source not in processed_files:
            try:
                blob = bucket.blob(f"{DATA_PATH}{os.path.basename(source)}")
                blob.delete()
                processed_files.add(source)
                processed_pdfs.append(os.path.basename(source))
                logger.info(f"Processed and deleted file: {source}")
            except Exception as e:
                logger.error(f"Error processing file {source}: {str(e)}")
                
    new_count = current_count + len(processed_files)
    with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as temp_file:
        temp_file.write(f"{new_count}\n")
        for pdf in processed_pdfs:
            temp_file.write(f"{pdf}\n")
        temp_file.flush()
        vectorized_blob.upload_from_filename(temp_file.name)
        os.unlink(temp_file.name)
        
    logger.info(f"Updated vectorized.txt with {len(processed_files)} new files. Total count: {new_count}")
    return len(processed_files)
