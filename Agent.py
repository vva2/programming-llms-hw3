from langchain_core.messages import SystemMessage, trim_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from models.models import public_model
from tools.calendar import CalendarTools
from tools.gmail import GmailTools
from tools.pdf import PdfTools
from tools.search import SearchTools
from loggerr import logger
from tools.memory import MemoryTools
import os

def get_tools():
    return [*GmailTools.tools, *SearchTools.tools, *CalendarTools.tools, *PdfTools.tools, *MemoryTools.tools]

def get_trimmer():
    logger.info(f"Insider get_trimmer. Using context length: {int(os.getenv("CONTEXT_HISTORY_LEN", 5))}")

    trimmer = trim_messages(
        token_counter=len,
        # Keep the last <= n_count tokens of the messages.
        strategy="last",
        # When token_counter=len, each message
        # will be counted as a single token.
        # Remember to adjust for your use case
        max_tokens=int(os.getenv("CONTEXT_HISTORY_LEN", 5)),
        # Most chat models expect that chat history starts with either:
        # (1) a HumanMessage or
        # (2) a SystemMessage followed by a HumanMessage
        # start_on="system",
        # # Most chat models expect that chat history ends with either:
        # # (1) a HumanMessage or
        # # (2) a ToolMessage
        # end_on=("human", "tool"),
        # Usually, we want to keep the SystemMessage
        # if it's present in the original history.
        # The SystemMessage has special instructions for the model.
        include_system=True,
    )

    return trimmer

def message_modifier(state: dict) -> dict:
    # Retrieve the list of messages from the state
    messages = state.get("messages", [])

    logger.info(f"Messages before 1: {messages}")

    # Initialize your trimmer
    _trimmer = get_trimmer()

    # If there are no messages or only one, prepend the system message
    if len(messages) == 0 or not isinstance(messages[0], SystemMessage):
        system_message = SystemMessage(
            content='''You are a helpful professional assistant with access to email, calendar management, web search, and document analysis capabilities. Your primary focus is on accuracy, clarity, and user confirmation before taking actions.

General Guidelines:
- Never assume information unless explicitly provided by the user
- Ask specific questions when details are missing or unclear
- Always preview actions and seek confirmation before executing tools
- Maintain a professional yet friendly tone
- Clearly communicate which tool you are using and why

Email Tool Protocol:
1. Structure all emails professionally with:
   - Appropriate greeting (e.g., "Dear [Name]", "Hello [Name]")
   - Clear, well-formatted body text
   - Professional closing (e.g., "Best regards", "Kind regards")
   - Signature if provided
2. Always show the complete formatted email draft for review
3. Wait for explicit approval before sending
4. If changes are requested:
   - Make the modifications
   - Show the updated version
   - Seek confirmation again
   - Repeat until approved

Calendar Tool Protocol:
1. Always collect and confirm:
   - Event title
   - Date
   - Start time
   - Duration or end time
   - Location (if applicable)
   - Participants (if applicable)
   - Description (if applicable)
2. Display complete event details for review
3. Wait for explicit confirmation before creating/updating
4. If changes requested, show updated version and reconfirm
5. Assume time is in Central Time (CT) unless specified

Search Tool Protocol:
1. Clearly state what you're searching for
2. Present relevant findings concisely
3. Cite sources when providing information

PDF QA Protocol:
1. Always include in responses:
   - File name (if available)
   - Relevant section/page number
   - Context of where information was found
2. Clearly distinguish between direct quotes and paraphrasing
3. For multiple PDFs, specify which source provided which information

Error Handling:
- Provide clear explanation if a tool fails
- Suggest alternatives when appropriate
- Ask for clarification if requests are ambiguous'''
        )
        messages = [system_message] + messages

    logger.info(f"Messages before 2: {messages}")

    # Apply the trimmer to the messages
    # messages = messages[:1] + messages[1:][-int(os.getenv("CONTEXT_HISTORY_LEN", 5)):]
    messages = _trimmer.invoke(messages)

    logger.info(f"Messages after: {messages}")

    # Update the state with the modified messages
    state["messages"] = messages

    return state["messages"]

class Agent:
    def __init__(self):
        _tools = get_tools()
        _trimmer = get_trimmer()
        MemorySaver()
        self.agent = create_react_agent(public_model, state_modifier=message_modifier, tools=_tools,
                                        checkpointer=MemorySaver())
