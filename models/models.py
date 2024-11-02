import os
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from loggerr import logger

LOCAL_MODEL_NAME = os.getenv('MISTRAL_NEMO_12B_MODEL')
PUBLIC_MODEL_NAME = os.getenv("ANTHROPIC_3_5_MODEL")


def get_local_model():
    return ChatOllama(model=LOCAL_MODEL_NAME, temperature=0)


logger.info(f"Using Fully config: {int(os.getenv('FULLY_LOCAL'))}")

if int(os.getenv('FULLY_LOCAL')) == 1:
    logger.info(f'Replacing public model with local model: {LOCAL_MODEL_NAME}')
    public_model = get_local_model()
else:
    logger.info(f'Using public model: {PUBLIC_MODEL_NAME}')
    public_model = ChatAnthropic(model_name=PUBLIC_MODEL_NAME, temperature=0)
