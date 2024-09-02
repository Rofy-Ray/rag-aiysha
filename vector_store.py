from functools import lru_cache
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = "data/db"

embedding_function = FastEmbedEmbeddings()

@lru_cache(maxsize=1)
def get_vector_store():
    if not os.path.exists(CHROMA_PATH):
        logger.warning(f"Chroma database not found at {CHROMA_PATH}. Creating a new one.")
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

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
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logger.info(f"Cleared database at {CHROMA_PATH}")
    else:
        logger.info(f"No database found at {CHROMA_PATH}")
