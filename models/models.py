import os

from dotenv import load_dotenv
from langchain.agents import create_react_agent
from langchain.chains import LLMChain
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama



class ModelFactory:
    LOCAL_MODEL_NAME = os.getenv('MISTRAL_NEMO_12B_MODEL')
    PUBLIC_MODEL_NAME = os.getenv("ANTHROPIC_3_5_MODEL")

    local_model = ChatOllama(model=LOCAL_MODEL_NAME, temperature=0)


    prompt_template = ChatPromptTemplate([
        ("system",
         "You are a helpful assistant with access to various tools. Before using a tool, confirm with the user if it would benefit them. For example, if sending an email, present the draft for approval first. Apply similar consideration to other tools, using your best judgment to ensure the user is informed before any action."),
        MessagesPlaceholder("messages")
    ])

    public_model = ChatAnthropic(model_name=PUBLIC_MODEL_NAME, temperature=0)
