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
from loggerr import logger

load_dotenv()

local_model = ChatOllama(model=os.getenv("MISTRAL_NEMO_12B_MODEL"), temperature=0)
model = ChatAnthropic(model_name=os.getenv("ANTHROPIC_3_5_MODEL"), temperature=0)
# model = ChatOllama(model=os.getenv("MISTRAL_NEMO_12B_MODEL"), temperature=0)
memory = MemorySaver()


class Email(BaseModel):
    """Email model"""
    recipient_email: str = Field(description="recipient email address")
    subject: str = Field(description="subject of the email")
    body: Optional[str] = Field(description="body of the email")





@tool
def send_email(recipient_email: str, subject: str, body: str):
    """sends an email. this function is called only when the user approves the draft email. call this function only when recipient email, subject, and body are available and the email is a valid email."""
    # """sends an email only when recipient and message is explicitly defined and the email is a valid email address."""

    logger.info('inside send_email')
    logger.info(f'recipient email: {recipient_email}')
    logger.info(f'subject: {subject}')
    logger.info(f'body: {body}')


    return "email sent!"

@tool
def create_draft_email(recipient_email: str, message: str):
    """creates a draft email before sending the email. recipient_email and message are explicitly provided and the email is a valid email."""
    logger.info('inside create_draft_email')
    logger.info(f'recipient email: {recipient_email}')
    logger.info(f'message: {message}')

    draft = model.with_structured_output(Email).invoke(f"recipient={recipient_email}, message={message}")
    missing_fields = []

    if draft.subject == '<UNKNOWN>':
        missing_fields.append('subject')
    if draft.body == '<UNKNOWN>':
        missing_fields.append('body')
    if draft.recipient_email == '<UNKNOWN>':
        missing_fields.append('recipient email')

    if missing_fields.__len__() > 0:
        return f"missing information: {missing_fields}. Please confirm additional info from the user."

    return f"generated draft: {model.with_structured_output(Email).invoke(f"recipient={recipient_email}, message={message}")}\n would you like to proceed with this draft?"

tools = [create_draft_email, send_email]
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


