import os
import logging
# import tempfile
import shutil
from google.cloud import storage
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUCKET_NAME = "aiysha-convos"
CHROMA_PATH = "database/"
DB_DIR = "chroma_db"
# LOCAL_TEMP_DIR = tempfile.mkdtemp()

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

LOCK_FILE = os.path.join(DB_DIR, "lock")

def acquire_lock():
    if os.path.exists(LOCK_FILE):
        logger.warning("Lock file already exists. Waiting for lock to be released.")
        while os.path.exists(LOCK_FILE):
            time.sleep(1)
    with open(LOCK_FILE, "w") as f:
        f.write("locked")

def release_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

@lru_cache(maxsize=1)
def get_embedding_function():
    try:
        logger.debug("Initializing FastEmbedEmbeddings")
        embedding_function = FastEmbedEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        logger.debug("FastEmbedEmbeddings initialized")
        
        logger.debug("Testing embed_query")
        test_embed = embedding_function.embed_query(text="test")
        logger.info(f"Embedding function initialized successfully. Test embedding shape: {len(test_embed)}")
        
        return embedding_function
    except Exception as e:
        logger.error(f"Error initializing embedding function: {str(e)}", exc_info=True)
        raise

def download_db_from_gcs(local_dir):
    blobs = bucket.list_blobs(prefix=CHROMA_PATH)
    for blob in blobs:
        local_path = os.path.join(local_dir, blob.name[len(CHROMA_PATH):])
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
    logger.info(f"Downloaded Chroma DB from GCS to {local_dir}")

def upload_db_to_gcs():
    for root, _, files in os.walk(DB_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, DB_DIR)
            blob = bucket.blob(os.path.join(CHROMA_PATH, relative_path))
            blob.upload_from_filename(local_path)
    logger.info(f"Uploaded Chroma DB to GCS from {DB_DIR}")

@lru_cache(maxsize=1)
def get_vector_store():
    acquire_lock()
    logger.debug("Entering get_vector_store")
    try:
        logger.debug("Getting embedding function")
        embedding_function = get_embedding_function()
        logger.debug("Embedding function retrieved")

        logger.debug(f"Checking for Chroma database in GCS at {CHROMA_PATH}chroma.sqlite3")
        if not bucket.blob(f"{CHROMA_PATH}chroma.sqlite3").exists():
            logger.warning(f"Chroma database not found in GCS at {CHROMA_PATH}. Creating a new one.")
            logger.debug(f"Initializing Chroma with persist_directory={DB_DIR}")
            db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
        else:
            logger.debug("Downloading existing database from GCS")
            download_db_from_gcs(DB_DIR)
            logger.debug(f"Initializing Chroma with persist_directory={DB_DIR}")
            db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
            
        db_file_path = os.path.join(DB_DIR, "chroma.sqlite3")
        if not os.path.exists(db_file_path):
            logger.error(f"Database file not found at {db_file_path}")
            raise FileNotFoundError(f"Database file not found at {db_file_path}")
        
        logger.debug("Vector store initialized successfully")
        return db
    except Exception as e:
        logger.error(f"Error in get_vector_store: {str(e)}", exc_info=True)
        return None
    finally:
        release_lock()

def add_to_chroma(chunks):
    db = get_vector_store()
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    logger.info(f"{len(existing_ids)} documents exist in vector store.")

    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        logger.info(f"Adding {len(new_chunks)} new documents to the vector store.")
        max_batch_size = 5000
        batches = [new_chunks[i:i+max_batch_size] for i in range(0, len(new_chunks), max_batch_size)]
        for batch in batches:
            batch_ids = [chunk.metadata["id"] for chunk in batch]
            db.add_documents(batch, ids=batch_ids)
        # new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        # db.add_documents(new_chunks, ids=new_chunk_ids)
        upload_db_to_gcs()
    else:
        logging.info("No new documents to add to the vector store.")

def query_vector_store(query: str, db, k: int = 3):
    # logger.info(f"Number of documents in the vector store: {db._collection.count()}")
    try:
        results = db.similarity_search_with_score(query, k=k)
        if not results:
            logger.warning("No results found for the given query.")
            return ""
        # logger.info(f"SIMILARITY SEARCH RESULTS: {results}")
        context = "\n\n".join([doc.page_content for doc, score in results])
        return context
    except Exception as e:
        logger.error(f"Error in query_vector_store: {str(e)}")
        raise

def clear_database():
    blobs = bucket.list_blobs(prefix=CHROMA_PATH)
    for blob in blobs:
        blob.delete()
    logger.info(f"Cleared database in GCS at {CHROMA_PATH}")
    shutil.rmtree(LOCAL_TEMP_DIR)
    logger.info(f"Cleared local temporary directory at {LOCAL_TEMP_DIR}")

# def cleanup():
#     if os.path.exists(LOCAL_TEMP_DIR):
#         shutil.rmtree(LOCAL_TEMP_DIR)
#         logger.info(f"Cleaned up temporary directory at {LOCAL_TEMP_DIR}")
#     else:
#         logger.info(f"Temporary directory {LOCAL_TEMP_DIR} does not exist, no cleanup needed")
