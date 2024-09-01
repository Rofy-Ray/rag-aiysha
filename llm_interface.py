from google.cloud import aiplatform
import os
import re
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT = os.getenv("PROJECT")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
LOCATION = os.getenv("LOCATION")
API_ENDPOINT = os.getenv("API_ENDPOINT")

SYSTEM_PROMPT = """<s>[INST]
<<SYS>>
You are a makeup artist and beauty advisor named Aiysha. You apply cosmetics on clients to enhance features, create looks and styles according to the latest trends in beauty and fashion. 
You offer advice about skincare routines, know how to work with different textures of skin tone, and are able to use both traditional methods and new techniques for applying products. 
Please respond with complete sentences and keep your responses under 280 characters.
<</SYS>>
"""

def get_llama_response(input_data):
    client_options = {"api_endpoint": API_ENDPOINT}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    endpoint = client.endpoint_path(
        project=PROJECT, location=LOCATION, endpoint=ENDPOINT_ID
    )
    instances = [
        {"inputs": input_data, 
         "parameters": {"max_tokens": 500, "temperature": 0}
        }
    ]
    response = client.predict(endpoint=endpoint, instances=instances)
    return response.predictions[0]

# def get_llama_response(input_data):
#     aiplatform.init(project=PROJECT, location=LOCATION)
#     endpoint = aiplatform.Endpoint(ENDPOINT_ID)
#     response = endpoint.predict(instances=[{"inputs": input_data}])
#     return response.predictions[0]

def format_llama_prompt(message: str, context: str) -> str:
    formatted_prompt = f"""{SYSTEM_PROMPT}\nGiven the following context, please answer the question. If the answer is not in the context, use your knowledge as Aiysha to provide a relevant response.\n\nContext: {context}\n\nQuestion: {message} [/INST]</s>"""
    return formatted_prompt

def get_model_response(message: str, context: str):
    query = format_llama_prompt(message, context)
    logger.info(f"FORMATTED PROMPT: {query}")
    generated_text = get_llama_response(query)
           
    if generated_text:
        response = generated_text
        logger.info(f"MODEL RESPONSE: {response}")
    else:
        response = "I'm sorry, I couldn't generate a response. Please try again."

    return response