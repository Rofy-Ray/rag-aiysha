import os
import logging
import tempfile
import shutil
from google.cloud import storage
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUCKET_NAME = "aiysha-convos"
CHROMA_PATH = "database/"
LOCAL_TEMP_DIR = tempfile.mkdtemp()

embedding_function = FastEmbedEmbeddings()
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def download_db_from_gcs():
    blobs = bucket.list_blobs(prefix=CHROMA_PATH)
    for blob in blobs:
        local_path = os.path.join(LOCAL_TEMP_DIR, blob.name[len(CHROMA_PATH):])
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
    logger.info(f"Downloaded Chroma DB from GCS to {LOCAL_TEMP_DIR}")

def upload_db_to_gcs():
    for root, _, files in os.walk(LOCAL_TEMP_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, LOCAL_TEMP_DIR)
            blob = bucket.blob(os.path.join(CHROMA_PATH, relative_path))
            blob.upload_from_filename(local_path)
    logger.info(f"Uploaded Chroma DB to GCS from {LOCAL_TEMP_DIR}")

@lru_cache(maxsize=1)
def get_vector_store():
    if not bucket.blob(f"{CHROMA_PATH}chroma.sqlite3").exists():
        logger.warning(f"Chroma database not found in GCS at {CHROMA_PATH}. Creating a new one.")
        db = Chroma(persist_directory=LOCAL_TEMP_DIR, embedding_function=embedding_function)
    else:
        download_db_from_gcs()
        db = Chroma(persist_directory=LOCAL_TEMP_DIR, embedding_function=embedding_function)
    return db

def add_to_chroma(chunks):
    db = get_vector_store()
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    logger.info(f"{len(existing_ids)} documents exist in vector store.")

    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        logger.info(f"Adding {len(new_chunks)} new documents to the vector store.")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        upload_db_to_gcs()
    else:
        logging.info("No new documents to add to the vector store.")

def query_vector_store(query: str, db, k: int = 3):
    logger.info(f"Number of documents in the vector store: {db._collection.count()}")
    results = db.similarity_search_with_score(query, k=k)
    if not results:
        logger.warning("No results found for the given query.")
        return ""
    logger.info(f"SIMILARITY SEARCH RESULTS: {results}")
    context = "\n\n".join([doc.page_content for doc, score in results])
    return context

def clear_database():
    blobs = bucket.list_blobs(prefix=CHROMA_PATH)
    for blob in blobs:
        blob.delete()
    logger.info(f"Cleared database in GCS at {CHROMA_PATH}")
    shutil.rmtree(LOCAL_TEMP_DIR)
    logger.info(f"Cleared local temporary directory at {LOCAL_TEMP_DIR}")

def cleanup():
    shutil.rmtree(LOCAL_TEMP_DIR)
    logger.info(f"Cleaned up temporary directory at {LOCAL_TEMP_DIR}")
