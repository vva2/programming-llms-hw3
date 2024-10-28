from pprint import pprint
from typing import Optional, Union, List
from typing_extensions import Annotated
import os
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, InjectedState
from dotenv import load_dotenv
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel, Field

llm = None
memory = MemorySaver()


class Email(BaseModel):
    """Email model"""

    recipient_email: str = Field(description="recipient email address")
    subject: str = Field(description="subject of the email")
    body: Optional[str] = Field(description="body of the email")

class Search(BaseModel):
    """Search internet model"""

    query: str = Field(description="search query", default=None)

class UnionModel(BaseModel):
    """stores the union of all the possible models available"""
    model: Union[Email, Search]


@tool
def send_email(recipient_email: str, message: str,  state: InjectedState):
    """sends an email given recipient email and message"""

    print(f"{memory.__getstate__()['messages']}")

    print("inside send_email")
    print(f"state: {state}")
    # print(f"recipient: {email.recipient_email}")
    # print(f"subject: {email.subject}")
    # print(f"body: {email.body}")

    # print(f'sending email to {recipient_email}')
    # print(f'subject:\n {subject}')
    # print(f'body:\n {body}')

    return "email send successfully"

@tool
def error_log(error_log: str):
    """Logs the error to the user."""
    print(f'error: {error_log}')

    return "DONE"

class State(AgentState):
    docs: List[str]


if __name__ == '__main__':
    load_dotenv()

    # model = ChatAnthropic(model_name="claude-3-sonnet-20240229", temperature=0)
    model = ChatAnthropic(model_name=os.getenv("ANTHROPIC_3_MODEL"), temperature=0)

    tools = [send_email]

    agent = create_react_agent(model, tools, state_schema=State, checkpointer=memory)

    # email = model.with_structured_output(UnionModel).invoke([SystemMessage("You are a friendly personal assistant. You need to answer concisely to every question. Do not assume any information unless I explicity or implicitly provide them. Ask questions if you are unsure about anything."), HumanMessage("send an email about approaching course deadline.")]).model
    # print(email)
    #
    # print(f"recipient: {email.recipient_email}")
    # print(f"subject: {email.subject}")
    # print(f"body: {email.body}")
    # print('-------')

    config = {"configurable": {"thread_id": "asdasd"}}

    docs = [
        "FooBar company just raised 1 Billion dollars!",
        "FooBar company was founded in 2019",
    ]

    inputs = {
        "messages": [{"type": "user", "content": "send an email to vva2@tamu.edu about course deadline."}],
        "docs": docs,
    }

    response = agent.invoke(inputs, config=config)

    pprint(f'response: {response}')
