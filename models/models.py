import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

load_dotenv()

class ModelFactory:
    LOCAL_MODEL_NAME = os.getenv('MISTRAL_NEMO_12B_MODEL')
    PUBLIC_MODEL_NAME = os.getenv("ANTHROPIC_3_5_MODEL")

    local_model = ChatOllama(model=LOCAL_MODEL_NAME, temperature=0)
    public_model = ChatAnthropic(model_name=PUBLIC_MODEL_NAME, temperature=0)
