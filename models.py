import os

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline


class Models:
    def __init__(self):
        pass

    def load_model(model_name: str):
        if model_name == os.getenv('NEXUSRAVEN_MODEL'):
            model = pipeline(
                "text-generation",
                model="Nexusflow/NexusRaven-13B",
                torch_dtype="auto",
                device_map="auto",
            )

            llm = HuggingFacePipeline(pipeline=model)

            return llm
        elif model_name == os.getenv('HERMES_2_PRO_LLAMA_3'):
            model = pipeline(
                "text-generation",
                model="Nexusflow/NexusRaven-13B",
                torch_dtype="auto",
                device_map="auto",
            )

            llm = HuggingFacePipeline(pipeline=model)

            return llm
        else:
            pass

