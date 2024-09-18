import streamlit as st
import os
import json
import logging
from PIL import Image
from llm_interface import get_model_response
from pdf_processor import process_new_pdfs
from vector_store import query_vector_store, get_vector_store, cleanup
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "pdf/new/"
BUCKET_NAME = "aiysha-convos"
CONVERSATION_TRACK_BLOB = "history/conversations.txt"
CHAT_HISTORY_BLOB = "history/chat_history.json"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vector_store = None

if not st.session_state.initialized:
    st.session_state.initialized = True
    st.session_state.vector_store = get_vector_store()

@st.cache_data
def load_chat_history():
    blob = bucket.blob(CHAT_HISTORY_BLOB)
    if blob.exists():
        return json.loads(blob.download_as_text())
    return []

@st.cache_data
def save_chat_history(messages):
    blob = bucket.blob(CHAT_HISTORY_BLOB)
    blob.upload_from_string(json.dumps(messages))

def check_and_process_new_pdfs():
    blobs = list(bucket.list_blobs(prefix=DATA_PATH))
    if not blobs:
        return False, ""
    processed_count = process_new_pdfs()
    if processed_count > 0:
        return True, f"I have {processed_count} beauty bot(s) busy blending and perfecting! Just like a good contour, it takes a little time to get it right."
    else:
        return False, ""
    
@st.cache_data
def load_image(image_path, size=(150, 150)):
    img = Image.open(image_path)
    return img

def update_conversation_count():
    blob = bucket.blob(CONVERSATION_TRACK_BLOB)
    
    if not blob.exists():
        blob.upload_from_string("queries: 0\nresponses: 0")
    
    try:
        content = blob.download_as_text()
        lines = content.split('\n')
        
        queries = int(lines[0].split(': ')[1])
        responses = int(lines[1].split(': ')[1])
        
        queries += 1
        responses += 1
        
        new_content = f"queries: {queries}\nresponses: {responses}"
        blob.upload_from_string(new_content)
    except Exception as e:
        logger.error(f"Error updating conversation count: {e}")
    
st.set_page_config(
    page_title="Aiysha from yShade.AI",
    page_icon="images/yshadelogobig.png",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None  
)

streamlit_style = """
			<style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
			</style>
		"""
   
st.markdown(streamlit_style, unsafe_allow_html=True)

with st.sidebar:
    logo = load_image("images/yshadelogo.png") 
    st.image(logo, use_column_width=True)

    st.header("About Me")
    st.write("""
    Hi, I'm Aiysha, your AI beauty assistant from yShade.AI, here to help with all your beauty-related questions.\n 
    Need steps to recreate a makeup look? I'm at your service!
    """)
    
    st.header("Coming Soon")
    st.write("""
    I’ll assist with virtual try-ons for hair, lipstick, and more.\n
    Get personalized product recommendations, like the perfect foundation and concealer for your skin tone.\n
    You can even chat with me via speech, and I’ll respond!
    """)

st.title("Aiysha from yShade.AI")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])
        cleanup()

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
            try:
                processing_status, message = check_and_process_new_pdfs()
                if processing_status:
                    st.info(f"Hold on. {message}")
            except Exception as e:
                logger.error(f"PDF processing error: {str(e)}", exc_info=True)
        
        message_placeholder = st.empty()
        context = query_vector_store(prompt, st.session_state.vector_store)
        logger.info(f"FETCHED CONTEXT: {context}")
        with st.spinner(""):
            response = get_model_response(prompt, context, st.session_state.messages)
        message_placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    
    update_conversation_count()

    save_chat_history(st.session_state.messages)