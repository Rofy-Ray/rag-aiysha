import streamlit as st
import os
import json
import logging
import json
import uuid
from PIL import Image
from llm_interface import get_text_response, get_image_response
from pdf_processor import process_new_pdfs
from vector_store import query_vector_store, get_vector_store
from dotenv import load_dotenv
from google.cloud import storage
from streamlit_session_browser_storage import SessionStorage
from audio_processor import transcribe_audio, text_to_speech

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "pdf/new/"
BUCKET_NAME = "aiysha-convos"
CONVERSATION_TRACK_BLOB = "history/conversations.txt"

bucket = storage.Client().bucket(BUCKET_NAME)

st.set_page_config(
    page_title="Aiysha from yShade.AI",
    page_icon="images/yshadeicon.png",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None  
)

session_storage = SessionStorage()

def get_chat_history_id():
    if "chat_history_id" not in st.session_state:
        chat_history_id = session_storage.getItem("chat_history_id")
        if not chat_history_id:
            chat_history_id = str(uuid.uuid4())
            session_storage.setItem("chat_history_id", chat_history_id) 
        st.session_state.chat_history_id = chat_history_id
    return st.session_state.chat_history_id

def save_chat_history(messages):
    chat_history_id = get_chat_history_id()
    blob = bucket.blob(f"history/{chat_history_id}.json")
    blob.upload_from_string(json.dumps(messages))

def load_chat_history():
    chat_history_id = get_chat_history_id()
    blob = bucket.blob(f"history/{chat_history_id}.json")
    if blob.exists():
        return json.loads(blob.download_as_string())
    return []

@st.cache_resource
def get_vector_store_wrapper():
    try:
        return get_vector_store()
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        return None

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

def upload_image_to_gcs(image_file):
    file_name = f"uploads/images/{uuid.uuid4()}.{image_file.name.split('.')[-1]}"
    blob = bucket.blob(file_name)
    blob.upload_from_file(image_file)
    gcs_uri = f"gs://{bucket.name}/{file_name}"
    public_url = blob.public_url
    return gcs_uri, public_url

def upload_audio_to_gcs(audio_bytes):
    file_name = f"uploads/audio/{uuid.uuid4()}.wav"
    blob = bucket.blob(file_name)
    blob.upload_from_string(audio_bytes, content_type='audio/wav')
    return blob.public_url

streamlit_style = """
            <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                .e1f1d6gn5 {
                    position: fixed;
                    bottom: 0;
                    width: 75%;
                    justify-content: center;
                    align-items: end;
                    margin-bottom: 0.5rem;
                }
                .e1f1d6gn5 .stFileUploader, .e1f1d6gn5 .stAudioInput {
                    margin-bottom: 3rem;
                }
                .e1f1d6gn5 .stTextInput, .e1f1d6gn5 .stButton {
                    position: absolute;
                    bottom: 0;
                }
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
    """)

st.title("Aiysha from yShade.AI")
USER_AVATAR = "👤"
BOT_AVATAR = Image.open("images/aiysha_avatar.png")

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("audio"):
            st.audio(message["audio"])
        else:
            if message.get("content"):
                st.markdown(message["content"])
            if message.get("image"):
                st.image(message["image"], width=200)

row_top = st.columns([1,1])
row_bottom = st.columns([9,1])

container = st.container()
with container:
    with row_top[0]:
        image_file = st.file_uploader(" ", type=["png", "jpg", "jpeg", "heic"], accept_multiple_files=False, label_visibility="collapsed", key="image_input")

    with row_top[1]:
        audio_input = st.experimental_audio_input(" ", label_visibility="collapsed", key="audio_input")

    with row_bottom[0]:
        prompt = st.text_input(" ", placeholder="My name is Aiysha! How can I assist you?", key="chat_input")

    with row_bottom[1]:
        send_button = st.button("", icon=":material/send:", disabled=not prompt and not audio_input)     

if send_button or audio_input:
    user_input = prompt
    public_image_url = None
    gcs_image_uri = None
    user_audio_url = None 

    if audio_input:
        audio_bytes = audio_input.read()
        try:
            transcription = transcribe_audio(audio_bytes)
            if transcription:
                user_input = transcription
                user_audio_url = upload_audio_to_gcs(audio_bytes)
        except Exception as e:
            logger.error(f"Error in ASR request: {str(e)}")
            st.error("An error occurred while processing your request. Please try again later.")
            st.stop()

    if image_file:
        gcs_image_uri, public_image_url = upload_image_to_gcs(image_file)

    if audio_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "audio": user_audio_url})
    else:
        st.session_state.messages.append({"role": "user", "content": user_input, "image": public_image_url})
    save_chat_history(st.session_state.messages)
    
    with st.chat_message("user", avatar=USER_AVATAR):
        if audio_input:
            st.audio(data=user_audio_url, format="audio/wav", start_time=0) 
        else:
            st.markdown(user_input)
            if public_image_url:
                st.image(public_image_url, width=200)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Thinking..."):
            try:
                processing_status, message = check_and_process_new_pdfs()
                if processing_status:
                    st.info(f"Hold on. {message}")
            except Exception as e:
                logger.error(f"PDF processing error: {str(e)}", exc_info=True)
                st.error("An error occurred while processing your request. Please try again later.")

        message_placeholder = st.empty()
        vector_store = get_vector_store_wrapper()
        if vector_store is not None:
            try:
                context = query_vector_store(user_input, vector_store)
                with st.spinner(""):
                    if gcs_image_uri:
                        response = get_image_response(user_input, gcs_image_uri, context, st.session_state.messages)
                    else:
                        response = get_text_response(user_input, context, st.session_state.messages)
                
                if audio_input:
                    try:
                        audio_response_url = text_to_speech(response)
                    except Exception as e:
                        logger.error(f"Error in TTS request: {str(e)}")
                        st.error("An error occurred while processing your request. Please try again later.")
                        st.stop()
                    st.session_state.messages.append({"role": "assistant", "content": response, "audio": audio_response_url}) 
                    save_chat_history(st.session_state.messages)
                    st.audio(data=audio_response_url, format="audio/wav", autoplay=True, start_time=0) 
                else:
                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    save_chat_history(st.session_state.messages)
                update_conversation_count()
            except Exception as e:
                logger.error(f"Error during query or response generation: {str(e)}", exc_info=True)
                st.error("An error occurred while processing your request. Please try again later.")
        else:
            logger.error("Unable to initialize the vector store. Please try again later.")
            st.error("An error occurred while processing your request. Please try again later.")