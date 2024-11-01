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
        logger.info(f'state before: {state}')

        state["messages"] = []

        logger.info(f'state after: {state}')

        return "cleared memory successfully"

    tools = [reset_memory]