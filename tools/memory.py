import os

from langchain_core.tools import tool
from langgraph.graph import MessagesState
from loggerr import logger

class MemoryTools:
    @tool
    def reset_memory(state: MessagesState):
        """
            Clears the message history in the current state.
        """
        logger.info("Resetting memory.")
        os.environ['RESET_MEMORY'] = 'YES'

        return "cleared memory successfully"

    tools = []
    # tools = [reset_memory]
