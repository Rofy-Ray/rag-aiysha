import os
# import re
import logging
import openai
import streamlit as st
# import threading
# import time
# from datetime import datetime, timedelta
# from typing import Optional, Dict, Tuple
# from google.cloud import aiplatform
# from google.protobuf import json_format
# from google.protobuf.struct_pb2 import Value
# from google.auth import default, transport
from google.auth.transport import requests
from google.auth import default
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# PROJECT = os.getenv("PROJECT")
# ENDPOINT_ID = os.getenv("ENDPOINT_ID")
# LOCATION = os.getenv("LOCATION")
# API_ENDPOINT = os.getenv("API_ENDPOINT")

MAAS_ENDPOINT = os.getenv("MAAS_ENDPOINT")
PROJECT_NUMBER = os.getenv("PROJECT_NUMBER")
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

#################################################################################################################################

# openai_client = None
# credentials_obj = None
# lock = threading.Lock()

# def refresh_credentials():
#     global credentials_obj
#     with lock:
#         credentials_obj, _ = default(scopes=SCOPES)
#         auth_request = transport.requests.Request()
#         credentials_obj.refresh(auth_request)
#     threading.Timer(3600.0, refresh_credentials).start()

# def get_openai_client(is_image_model=False):
#     global openai_client
#     global credentials_obj
#     with lock:
#         if credentials_obj is None:
#             refresh_credentials()
#         base_url = f"https://{MAAS_ENDPOINT}/v1beta1/projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/"
#         if is_image_model:
#             base_url += "openapi"
#         else:
#             base_url += "openapi/chat/completions?"
#         openai_client = openai.OpenAI(
#             base_url=base_url,
#             api_key=credentials_obj.token
#         )
#     return openai_client

#################################################################################################################################

# class OpenAIClientManager:
#     def __init__(self, project_number: str, maas_endpoint: str, scopes: list):
#         self._project_number = project_number
#         self._maas_endpoint = maas_endpoint
#         self._scopes = scopes
#         self._lock = threading.Lock()
        
#         self._credentials = None
#         self._credentials_expiry: Optional[datetime] = None
#         self._clients: Dict[bool, Tuple[openai.OpenAI, datetime]] = {}
        
#         self._refresh_thread: Optional[threading.Thread] = None
#         self._should_stop = threading.Event()
        
#         self._REFRESH_BUFFER_SECONDS = 300
#         self._MAX_RETRY_ATTEMPTS = 3
#         self._RETRY_DELAY_SECONDS = 5

#     def _refresh_credentials(self) -> None:
#         """Refresh credentials if they're expired or close to expiring."""
#         try:
#             with self._lock:
#                 current_time = datetime.now()
                
#                 if (self._credentials_expiry is None or 
#                     current_time + timedelta(seconds=self._REFRESH_BUFFER_SECONDS) >= self._credentials_expiry):
                    
#                     for attempt in range(self._MAX_RETRY_ATTEMPTS):
#                         try:
#                             logger.debug("Refreshing credentials...")
#                             self._credentials, _ = default(scopes=self._scopes)
#                             auth_request = requests.Request()
#                             self._credentials.refresh(auth_request)
                            
#                             self._credentials_expiry = datetime.now() + timedelta(seconds=3600)
                            
#                             self._clients.clear()
                            
#                             logger.info("Credentials refreshed successfully")
#                             break
                            
#                         except Exception as e:
#                             if attempt == self._MAX_RETRY_ATTEMPTS - 1:
#                                 logger.error(f"Failed to refresh credentials after {self._MAX_RETRY_ATTEMPTS} attempts: {e}")
#                                 raise
#                             else:
#                                 logger.warning(f"Refresh attempt {attempt + 1} failed: {e}")
#                                 time.sleep(self._RETRY_DELAY_SECONDS)
#         except Exception as e:
#             logger.error(f"Error refreshing credentials: {e}")
            
#     def _refresh_loop(self) -> None:
#         """Background loop to periodically refresh credentials."""
#         while not self._should_stop.is_set():
#             try:
#                 self._refresh_credentials()
#                 sleep_time = (self._credentials_expiry - datetime.now()).total_seconds() - self._REFRESH_BUFFER_SECONDS
#                 if sleep_time > 0:
#                     self._should_stop.wait(sleep_time)
#             except Exception as e:
#                 logger.error(f"Error in refresh loop: {e}")
#                 time.sleep(self._RETRY_DELAY_SECONDS)

#     def start(self) -> None:
#         """Start the credential refresh background thread."""
#         if self._refresh_thread is None or not self._refresh_thread.is_alive():
#             self._should_stop.clear()
#             self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
#             self._refresh_thread.start()
#             logger.info("Credential refresh thread started")

#     def stop(self) -> None:
#         """Stop the credential refresh background thread."""
#         self._should_stop.set()
#         if self._refresh_thread and self._refresh_thread.is_alive():
#             self._refresh_thread.join()
#             logger.info("Credential refresh thread stopped")

#     def get_client(self, is_image_model: bool = False) -> openai.OpenAI:
#         """Get an OpenAI client with current credentials."""
#         with self._lock:
#             self._refresh_credentials()
            
#             if is_image_model in self._clients:
#                 client, expiry = self._clients[is_image_model]
#                 if datetime.now() < expiry:
#                     return client
            
#             base_url = (f"https://{self._maas_endpoint}/v1beta1/projects/"
#                        f"{self._project_number}/locations/us-central1/endpoints/")
#             base_url += "openapi" if is_image_model else "openapi/chat/completions?"
            
#             client = openai.OpenAI(
#                 base_url=base_url,
#                 api_key=self._credentials.token
#             )
            
#             self._clients[is_image_model] = (client, self._credentials_expiry)
            
#             return client

#################################################################################################################################

# @st.cache_resource
# def get_ai_client():
#     client_options = {"api_endpoint": API_ENDPOINT}
#     return aiplatform.gapic.PredictionServiceClient(client_options=client_options)

# @st.cache_resource
# def get_openai_client():
#     credentials, _ = default()
#     auth_request = transport.requests.Request()
#     credentials.refresh(auth_request)
#     return openai.OpenAI(
#         base_url=f"https://{MAAS_ENDPOINT}/v1beta1/projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/openapi",
#         api_key=credentials.token
#     )
    
# @st.cache_resource(ttl=3600)

def get_openai_client(is_image_model=False):
    credentials, _ = default(scopes=SCOPES)
    auth_request = requests.Request()
    credentials.refresh(auth_request)
    
    base_url = f"https://{MAAS_ENDPOINT}/v1beta1/projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/"
    if is_image_model:
        base_url += "openapi"
    else:
        base_url += "openapi/chat/completions?"
    
    return openai.OpenAI(
        base_url=base_url,
        api_key=credentials.token
    )


# SYSTEM_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a makeup artist and beauty advisor named Aiysha. You apply cosmetics on clients to enhance features, create looks and styles according to the latest trends in beauty and fashion. 
# You offer advice about skincare routines, know how to work with different textures of skin tone, and are able to use both traditional methods and new techniques for applying products. 
# Please respond with complete sentences and keep your responses under 280 characters.
# <|eot_id|>"""

# @st.cache_data
# def format_llama_prompt(message: str, context: str, chat_history: list) -> str:
#     formatted_prompt = SYSTEM_PROMPT

#     for turn in chat_history:
#         role = "user" if turn["role"] == "user" else "assistant"
#         formatted_prompt += f"<|start_header_id|>{role}<|end_header_id|>{turn['content']}<|eot_id|>"
    
#     formatted_prompt += f"""<|start_header_id|>user<|end_header_id|>Given the following context, please answer the question. If the answer is not in the context, use your knowledge as Aiysha to provide a relevant response.

#     Context: {context}

#     Question: {message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

#     return formatted_prompt
    
# def get_text_response(message: str, context: str, chat_history: list):
#     query = format_llama_prompt(message, context, chat_history)
#     # logger.info(f"FORMATTED PROMPT: {query}")
    
#     client = get_ai_client()
#     endpoint = client.endpoint_path(project=PROJECT, location=LOCATION, endpoint=ENDPOINT_ID)
    
#     # instances = [{"inputs": query, "parameters": {"max_tokens": 4096, "temperature": 0}}]
#     # response = client.predict(endpoint=endpoint, instances=instances)
    
#     instances = [{"prompt": query}]
#     response = client.predict(endpoint=endpoint, instances=instances, parameters={"max_new_tokens": 2048, "temperature": 0, "truncation": False, "return_full_text": True})
    
#     logger.info(f"MODEL RESPONSE: {response}") 
    
#     if response.predictions:
#         output_text = response.predictions[0]
#         pattern = r"Output:\s*(.*)$"
#         match = re.search(pattern, output_text, re.DOTALL)
#         if match:
#             generated_text = match.group(1).strip()
#         else:
#             generated_text = None
#     else:
#         generated_text = None
        
#     # generated_text = response.predictions[0][0] if response.predictions else None
#     # generated_text = response.predictions[0] if response.predictions else None
    
#     if generated_text:        
#         # logger.info(f"MODEL OUTPUT: {generated_text}")
#         return generated_text
#     else:
#         return "I'm sorry, I couldn't generate a response. Please try again."

def get_text_response(message: str, context: str, chat_history: list):
    client = get_openai_client(is_image_model=False)
    
    # client_manager = OpenAIClientManager(PROJECT_NUMBER, MAAS_ENDPOINT, SCOPES)
    # client_manager.start()
    
    # try: 
    #     logger.info("Creating chat client")
    #     chat_client = client_manager.get_client(is_image_model=False)
    
    system_message = """
    You are a makeup artist and beauty advisor named Aiysha. You apply cosmetics on clients to enhance features, create looks and styles according to the latest trends in beauty and fashion. 
    You offer advice about skincare routines, know how to work with different textures of skin tone, and are able to use both traditional methods and new techniques for applying products. 
    Please respond with complete sentences and keep your responses under 280 characters.
    """
    
    messages = [
        {"role": "system", "content": system_message},
    ]

    for turn in chat_history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user", 
        "content": f"Context: {context}\n\nQuestion: {message}"
    })
    
    try:
        response = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct-maas",
            messages=messages,
            max_tokens=4096,
        )
        
        generated_content = response.choices[0].message.content if response.choices else None
        
        if generated_content:
            return generated_content
        else:
            logger.warning("Empty response from text model")
            return "I'm sorry, I couldn't generate a response. Could you please try rephrasing your question?"
    
    except Exception as e:
        logger.error(f"Error in text response generation: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again."
    
    # finally:
    #     logger.info("Closing OpenAI client")
    #     client_manager.stop()
    
def get_image_response(message: str, image_url: str, context: str, chat_history: list):
    client = get_openai_client(is_image_model=True)
    
    system_message = """ 
    You are a makeup artist and beauty advisor named Aiysha. 
    You apply cosmetics on clients to enhance features, create looks and styles according to the latest trends in beauty and fashion. 
    You offer advice about skincare routines, know how to work with different textures of skin tone, and are able to use both traditional methods and new techniques for applying products.
    Analyze the image and answer the question based on your expertise and context provided.
    """
    
    messages = [
        {"role": "system", "content": system_message},
    ]

    for turn in chat_history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user", 
        "content": [
            {"image_url": {"url": image_url}, "type": "image_url"},
            {"text": f"Context: {context}\n\nQuestion: {message}", "type": "text"},
        ]
    })
    
    try:
        response = client.chat.completions.create(
            model="meta/llama-3.2-90b-vision-instruct-maas",
            messages=messages,
            max_tokens=4096,
        )
        
        generated_content = response.choices[0].message.content if response.choices else None
        
        if generated_content:
            return generated_content
        else:
            logger.warning("Empty response from image model")
            return "I'm sorry, I couldn't generate a response based on the image. Could you please try rephrasing your question or uploading a different image?"
    
    except Exception as e:
        logger.error(f"Error in image response generation: {str(e)}")
        return "I apologize, but I encountered an error while processing the image. Please try again or consider using a different image."