import os
import logging
import openai
import streamlit as st
from google.auth.transport import requests
from google.auth import default
from dotenv import load_dotenv
# from google.cloud import aiplatform

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# PROJECT = os.getenv("PROJECT")
# ENDPOINT_ID = os.getenv("ENDPOINT_ID")
# LOCATION = os.getenv("LOCATION")
# API_ENDPOINT = os.getenv("API_ENDPOINT")

# @st.cache_resource
# def get_ai_client():
#     client_options = {"api_endpoint": API_ENDPOINT}
#     return aiplatform.gapic.PredictionServiceClient(client_options=client_options)

MAAS_ENDPOINT = os.getenv("MAAS_ENDPOINT")
PROJECT_NUMBER = os.getenv("PROJECT_NUMBER")
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

def get_openai_client(is_image_model=False):
    credentials, _ = default(scopes=SCOPES)
    auth_request = requests.Request()
    credentials.refresh(auth_request)
    
    endpoint = "openapi" if is_image_model else "openapi/chat/completions"
    base_url = f"https://{MAAS_ENDPOINT}/v1beta1/projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/{endpoint}"
    
    if not is_image_model:
        base_url += "?"
    
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
    
def get_image_response(message: str, image_url: str, context: str, chat_history: list):
    client = get_openai_client(is_image_model=True)
    
    system_message = """ 
    You are a makeup artist and beauty advisor named Aiysha. 
    You apply cosmetics on clients to enhance features, create looks and styles according to the latest trends in beauty and fashion. 
    You offer advice about skincare routines, know how to work with different textures of skin tone, and are able to use both traditional methods and new techniques for applying products.
    Analyze the image and answer the question based on your expertise and context provided.
    """
    
    messages = [
        {
            "role": "user", 
            "content": [
                {"image_url": {"url": image_url}, "type": "image_url"},
                {"text": f"Context: {context}\n\nQuestion: {message}", "type": "text"},
            ]
        },
        {"role": "assistant", "content": system_message},
    ]

    for turn in chat_history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    
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