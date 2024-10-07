import requests
import logging
import os
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ASR_ENDPOINT = os.getenv("ASR_ENDPOINT")
TTS_ENDPOINT = os.getenv("TTS_ENDPOINT")

def transcribe_audio(audio_file):
    try:
        files = {'file': ('audio.wav', audio_file, 'audio/wav')}
        response = requests.post(ASR_ENDPOINT, files=files)
        response.raise_for_status()
        result = response.json()
        return result['text']
    except requests.RequestException as e:
        logger.error(f"Error in ASR request: {str(e)}")
        return None

def text_to_speech(text):
    try:
        payload = {"text": text}
        response = requests.post(TTS_ENDPOINT, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['audio_url']
    except requests.RequestException as e:
        logger.error(f"Error in TTS request: {str(e)}")
        return None