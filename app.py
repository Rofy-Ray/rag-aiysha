import streamlit as st
import os
import shelve
import logging
from llm_interface import get_model_response
from pdf_processor import process_new_pdfs
from vector_store import query_vector_store
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data/pdf/new"

@st.cache_resource
def get_vector_store():
    from vector_store import get_vector_store
    return get_vector_store()

@st.cache_data
def load_chat_history():
    with shelve.open("data/history/chat_history") as db:
        return db.get("messages", [])

@st.cache_data
def check_and_process_new_pdfs():
    if not os.listdir(DATA_PATH):
        return False, ""
    processed_count = process_new_pdfs()
    if processed_count > 0:
        return True, f"I have {processed_count} beauty bot(s) busy blending and perfecting! Just like a good contour, it takes a little time to get it right."
    else:
        return False, ""
    
st.set_page_config(page_title="Aiysha from yShade.AI", layout="wide")

st.title("Aiysha from yShade.AI")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        with shelve.open("data/history/chat_history") as db:
            db["messages"] = []

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("My name is Aiysha! How can I assist you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Thinking..."):
            processing_status, message = check_and_process_new_pdfs()
            if processing_status:
                st.info(f"Hold on. {message}")
        
        message_placeholder = st.empty()
        vector_store = get_vector_store()
        context = query_vector_store(prompt, vector_store)
        logger.info(f"FETCHED CONTEXT: {context}")
        with st.spinner(""):
            response = get_model_response(prompt, context)
        message_placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

with shelve.open("data/history/chat_history") as db:
    db["messages"] = st.session_state.messages