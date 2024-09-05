import streamlit as st
import os
import shelve
import logging
from PIL import Image
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
    
@st.cache_data
def load_image(image_path, size=(150, 150)):
    img = Image.open(image_path)
    # img = img.resize(size)
    return img
    
st.set_page_config(
    page_title="Aiysha from yShade.AI",
    page_icon="images/yshadelogobig.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None  
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    logo = load_image("images/yshadelogo.png") 
    st.image(logo, use_column_width=True)

    st.header("About Me")
    st.write("""
    Hi, I'm Aiysha, your AI beauty assistant from yShade.AI.\n 
    I'm here to help you with all your beauty-related questions and needs.\n 
    Do you want the steps to recreate a particular makeup look? I'm at your service!
    """)
    
    st.header("Coming Soon")
    st.write("""
    I will help you with virtual try-on (VTO) for hair, lipstick/lip gloss and others.\n 
    I will provide you with custom product recommendations like finding the perfect shade of foundation, concealer and other makeup products based on your unique skin shade and undertones.\n
    You can chat with me via speech and I will talk back to you.
    """)

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
            try:
                processing_status, message = check_and_process_new_pdfs()
                if processing_status:
                    st.info(f"Hold on. {message}")
            except Exception as e:
                logger.error(f"PDF processing error: {str(e)}", exc_info=True)
        
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