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
        if blob.name.lower().endswith('.pdf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                try:
                    logger.info(f"Downloading blob: {blob.name}")
                    blob.download_to_filename(temp_file.name)
                    
                    if os.path.getsize(temp_file.name) == 0:
                        logger.warning(f"Empty file downloaded: {blob.name}")
                        continue
                    
                    logger.info(f"Loading PDF: {temp_file.name}")
                    loader = PyPDFLoader(temp_file.name)
                    pdf_documents = loader.load()
                    
                    if not pdf_documents:
                        logger.warning(f"No content extracted from PDF: {blob.name}")
                    else:
                        for doc in pdf_documents:
                            doc.metadata["source"] = blob.name
                        documents.extend(pdf_documents)
                        logger.info(f"Successfully processed PDF: {blob.name}")
                
                except Exception as e:
                    logger.error(f"Error processing {blob.name}: {str(e)}")
                
                finally:
                    try:
                        os.unlink(temp_file.name)
                    except Exception as e:
                        logger.error(f"Error deleting temporary file {temp_file.name}: {str(e)}")
    
    logger.info(f"Total documents loaded: {len(documents)}")
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
    
    unique_sources = set(doc.metadata.get("source") for doc in documents if doc.metadata.get("source"))
    
    for source in unique_sources:
        if source.startswith(DATA_PATH):
            try:
                blob = bucket.blob(source)
                if blob.exists():
                    blob.delete()
                    processed_files.add(source)
                    processed_pdfs.append(os.path.basename(source))
                    logger.info(f"Processed and deleted file: {source}")
                else:
                    logger.warning(f"File not found in bucket: {source}")
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