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
from pydantic import BaseModel, Field

from models.models import ModelFactory
from tools.gmail import GmailTool
from tools.search import SearchTool
from utils.loggerr import logger



local_model = ModelFactory.local_model
model = ModelFactory.public_model
memory = MemorySaver()

tools = [GmailTool.send_email, SearchTool.search]
tool_node = ToolNode(tools)

checkpointer = MemorySaver()
agent = create_react_agent(model, tools, checkpointer=checkpointer)

class PrivateInfoFlag(BaseModel):
    has_private_info: bool = Field(description="true if private data is present, false otherwise")

def can_proceed_safely(user_input: str) -> bool:
    private_data_check_response = local_model.invoke(
        [
            HumanMessage(content=f"Identify the private information in this user prompt. If the private information is not explicitly mentioned then simply dont report anything.:\n-----------`{user_input}`\n-----------")
        ]
    )

    prompt = f"The following response describes the private data found in an user prompt. Using this data answer 'Yes' (if private data is indeed present) or 'No' (otherwise or if you are unsure). BE CONCISE I DO NOT NEED ANY EXPLANATION.\n-----\n{private_data_check_response.content}\n-----"

    result = local_model.invoke([HumanMessage(prompt)])

    logger.info(f'response: {result} for prompt: {prompt}')

    if result and 'yes' in result.content.lower():
        print(f"\n!!PRIVATE INFO FOUND!!:\n{private_data_check_response.content}")
        user_consent = input("Do you want to proceed sending this info to a public LLM? (y/n): ")

        if user_consent.lower().strip() != 'y':
            print(f"Your most recent input has been cleared from memory. Please re-phrase it for me.")
            return False

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


