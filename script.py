from pprint import pprint
from typing import Optional, Union, List

from langchain_ollama import ChatOllama
from typing_extensions import Annotated
import os
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, InjectedState, ToolNode
from dotenv import load_dotenv
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from utils.loggerr import logger

load_dotenv()

LOCAL_MODEL_NAME = os.getenv('MISTRAL_NEMO_12B_MODEL')
PUBLIC_MODEL_NAME = os.getenv("ANTHROPIC_3_5_MODEL")

local_model = ChatOllama(model=LOCAL_MODEL_NAME, temperature=0)
model = ChatAnthropic(model_name=PUBLIC_MODEL_NAME, temperature=0)
# model = ChatOllama(model=os.getenv("MISTRAL_NEMO_12B_MODEL"), temperature=0)
memory = MemorySaver()


class Email(BaseModel):
    """Email model"""
    recipient_email: str = Field(description="recipient email address")
    subject: str = Field(description="subject of the email")
    body: str = Field(description="body of the email including the sender signature.")





@tool
def send_email(recipient_email: str, message: str):
    """sends an email. call this function only when recipient email, and message are explicitly provided by the user and the recipient_email is a valid email."""

    draft = model.with_structured_output(Email).invoke(f"recipient={recipient_email}, message={message}")
    missing_fields = []

    if draft.subject == '<UNKNOWN>':
        missing_fields.append('subject')
    if draft.body == '<UNKNOWN>':
        missing_fields.append('body')
    if draft.recipient_email == '<UNKNOWN>':
        missing_fields.append('recipient email')

    if missing_fields.__len__() > 0:
        return f"missing information: {missing_fields}. Please confirm additional info from the user. Here are the details of the draft: {draft.__dict__}"


    print(f"Sending the email with the following info:\n{draft.__dict__}\n")

    user_input = input("Would you like to proceed? (y/n) ")

    if user_input.strip().lower() != 'y':
        return f"Sender (User) denied the send email request. May not be pleased with the draft. Ask the user the reason for denying the request."

    logger.info('inside send_email')
    logger.info(f'recipient email: {draft.recipient_email}')
    logger.info(f'subject: {draft.subject}')
    logger.info(f'body: {draft.body}')

    return "email sent!"

# @tool
# def create_draft_email(recipient_email: str, message: str):
#     """creates a draft email before sending the email. recipient_email and message are explicitly provided and the email is a valid email."""
#     logger.info('inside create_draft_email')
#     logger.info(f'recipient email: {recipient_email}')
#     logger.info(f'message: {message}')
#
#     return f"generated draft: {model.with_structured_output(Email).invoke(f"recipient={recipient_email}, message={message}")}\n would you like to proceed with this draft?"


tools = [send_email]
tool_node = ToolNode(tools)

checkpointer = MemorySaver()
agent = create_react_agent(model, tools, checkpointer=checkpointer)

class PrivateInfo(BaseModel):
    has_private_info: bool = Field(description="true if private data is present, false otherwise")
    details: str = Field(description="description of the private data")

class PrivateInfoFlag(BaseModel):
    has_private_info: bool = Field(description="true if private data is present, false otherwise")

@tool
def read_pdf(file_path: str):
    """reads a pdf given a file path"""

    return "read pdf successfully"

def can_proceed_safely(user_input: str) -> bool:
    # resp = local_model.with_structured_output(PrivateInfo).invoke([HumanMessage("Identify the private information in this user prompt. If the private information is not explicitly mentioned then simply dont report anything.:\n-----------`{user_input}`\n-----------")])

    private_data_check_response = local_model.invoke(
        [
            HumanMessage(content=f"Identify the private information in this user prompt. If the private information is not explicitly mentioned then simply dont report anything.:\n-----------`{user_input}`\n-----------")
        ]
    )

    prompt = f"The following response describes the private data found in an user prompt. Using this response answer 'Yes' (if private data is indeed present) or 'No' (otherwise)\n-----\n{private_data_check_response.content}\n-----"

    result = local_model.with_structured_output(PrivateInfoFlag).invoke(prompt)

    if result is None:
        logger.error(f"result is none in {can_proceed_safely} for prompt: {prompt}")
        logger.info("Skipping the private information check")
        return True

    if result.has_private_info:
        print(f"\n!!PRIVATE INFO FOUND!!:\n{private_data_check_response.content}")
        user_consent = input("Do you want to proceed sending this info to a public LLM? (y/n): ")

        if user_consent.lower().strip() != 'y':
            print(f"Your most recent input has been cleared from memory. Please re-phrase it for me.")
            return False

        return True


    return True


if __name__ == '__main__':
    while True:
        try:
            config = {"configurable": {"thread_id": 53}}
            user_input = input("USER>\n")

            if len(user_input) == 0 or not can_proceed_safely(user_input):
                continue

            response = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

            logger.info(response)

            print(f'AGENT>\n{response['messages'][-1].content}\n')
        except KeyboardInterrupt:
            break


