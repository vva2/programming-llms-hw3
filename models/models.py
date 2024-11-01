import os
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama


class ModelFactory:
    LOCAL_MODEL_NAME = os.getenv('MISTRAL_NEMO_12B_MODEL')
    PUBLIC_MODEL_NAME = os.getenv("ANTHROPIC_3_5_MODEL")

    @staticmethod
    def get_local_model():
        return ChatOllama(model=ModelFactory.LOCAL_MODEL_NAME, temperature=0)

    if os.getenv('FULLY_LOCAL') == 1:
        public_model = get_local_model()
    else:
        public_model = ChatAnthropic(model_name=PUBLIC_MODEL_NAME, temperature=0)
