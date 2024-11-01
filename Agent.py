from langchain_core.messages import SystemMessage, trim_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from models.models import ModelFactory
from tools.calendar import CalendarTools
from tools.gmail import GmailTools
from tools.pdf import PdfTools
from tools.search import SearchTools

def get_tools():
    return [*GmailTools.tools, *SearchTools.tools, *CalendarTools.tools, *PdfTools.tools]

def get_trimmer():
    trimmer = trim_messages(
        token_counter=len,
        # Keep the last <= n_count tokens of the messages.
        strategy="last",
        # When token_counter=len, each message
        # will be counted as a single token.
        # Remember to adjust for your use case
        max_tokens=10,
        # Most chat models expect that chat history starts with either:
        # (1) a HumanMessage or
        # (2) a SystemMessage followed by a HumanMessage
        start_on="human",
        # Most chat models expect that chat history ends with either:
        # (1) a HumanMessage or
        # (2) a ToolMessage
        end_on=("human", "tool"),
        # Usually, we want to keep the SystemMessage
        # if it's present in the original history.
        # The SystemMessage has special instructions for the model.
        include_system=True,
    )

    return trimmer

def message_modifier(state: dict) -> dict:
    # Retrieve the list of messages from the state
    messages = state.get("messages", [])

    # Initialize your trimmer
    _trimmer = get_trimmer()

    # If there are no messages or only one, prepend the system message
    if len(messages) <= 1:
        system_message = SystemMessage(
            content=(
                "You are a helpful assistant with access to various tools. "
                "Before using a tool, confirm with the user if it would benefit them. "
                "For example, if sending an email, present the draft for approval first. "
                "Apply similar consideration to other tools, using your best judgment to ensure the user is informed before any action."
            )
        )
        messages = [system_message] + messages

    # Apply the trimmer to the messages
    messages = _trimmer.invoke(messages)

    # Update the state with the modified messages
    state["messages"] = messages

    return state["messages"]

class Agent:
    def __init__(self):
        _tools = get_tools()
        _trimmer = get_trimmer()
        self.agent = create_react_agent(ModelFactory.public_model, state_modifier=message_modifier, tools=_tools, checkpointer=MemorySaver())



