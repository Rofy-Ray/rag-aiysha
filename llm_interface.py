import os
import re
import logging
import streamlit as st
from google.cloud import aiplatform
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT = os.getenv("PROJECT")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
LOCATION = os.getenv("LOCATION")
API_ENDPOINT = os.getenv("API_ENDPOINT")

@st.cache_resource
def get_ai_client():
    client_options = {"api_endpoint": API_ENDPOINT}
    return aiplatform.gapic.PredictionServiceClient(client_options=client_options)


SYSTEM_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a makeup artist and beauty advisor named Aiysha. You apply cosmetics on clients to enhance features, create looks and styles according to the latest trends in beauty and fashion. 
You offer advice about skincare routines, know how to work with different textures of skin tone, and are able to use both traditional methods and new techniques for applying products. 
Please respond with complete sentences and keep your responses under 280 characters.
<|eot_id|>"""

@st.cache_data
def format_llama_prompt(message: str, context: str, chat_history: list) -> str:
    formatted_prompt = SYSTEM_PROMPT

    for turn in chat_history:
        role = "user" if turn["role"] == "user" else "assistant"
        formatted_prompt += f"<|start_header_id|>{role}<|end_header_id|>{turn['content']}<|eot_id|>"
    
    formatted_prompt += f"""<|start_header_id|>user<|end_header_id|>Given the following context, please answer the question. If the answer is not in the context, use your knowledge as Aiysha to provide a relevant response.

    Context: {context}

    Question: {message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    return formatted_prompt
    
def get_model_response(message: str, context: str, chat_history: list):
    query = format_llama_prompt(message, context, chat_history)
    logger.info(f"FORMATTED PROMPT: {query}")
    
    client = get_ai_client()
    endpoint = client.endpoint_path(project=PROJECT, location=LOCATION, endpoint=ENDPOINT_ID)
    instances = [{"inputs": query, "parameters": {"max_tokens": 500, "temperature": 0}}]
    
    response = client.predict(endpoint=endpoint, instances=instances)
    generated_text = response.predictions[0] if response.predictions else None
    
    if generated_text:
        logger.info(f"MODEL RESPONSE: {generated_text}")
        return generated_text
    else:
        return "I'm sorry, I couldn't generate a response. Please try again."