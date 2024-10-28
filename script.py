from pprint import pprint
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv


@tool
def send_email(recipient_email: Optional[str] = None, subject: Optional[str] = None, body: Optional[str] = None):
    """sends an email to the 'recipient_email' with the information like subject and body inferred from the user prompt. Note that if the any information is unclear this function should not run."""
    print(f'sending email to {recipient_email}')
    print(f'subject:\n {subject}')
    print(f'body:\n {body}')

    return "email send successfully"


if __name__ == '__main__':
    load_dotenv()

    memory = MemorySaver()
    model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
    search = TavilySearchResults(max_results=1)
    tools = [search, send_email]
    model_with_tools = create_react_agent(model, tools, checkpointer=memory)

    config = {"configurable": {"thread_id": "abc123"}}

    response = model_with_tools.invoke({
        "messages": [HumanMessage(content="send email to vva2@tamu.edu")],
    }, config=config)

    pprint(response['messages'][-1].content)

