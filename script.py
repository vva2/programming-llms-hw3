import os

from dotenv import load_dotenv
load_dotenv()

from Agent import Agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from models.models import ModelFactory
from utils.loggerr import logger


class PrivateInfoFlag(BaseModel):
    has_private_info: bool = Field(description="true if private data is present, false otherwise")

def can_proceed_safely(user_input: str, local_model) -> bool:
    private_data_check_response = local_model.invoke(
        [
            HumanMessage(content=f"Identify the private information in this user prompt. If the private information is not explicitly mentioned then simply dont report anything.:\n-----------`{user_input}`\n-----------")
        ]
    )

    prompt = f"The following response describes the private data found in an user prompt. Using this data answer 'Yes' (if private data is indeed present) or 'No' (otherwise or if you are unsure). BE CONCISE I DO NOT NEED ANY EXPLANATION.\n-----\n{private_data_check_response.content}\n-----"

    result = ModelFactory.local_model.invoke([HumanMessage(prompt)])

    logger.info(f'response: {result} for prompt: {prompt}')

    if result and 'yes' in result.content.lower():
        print(f"\n!!PRIVATE INFO FOUND!!:\n{private_data_check_response.content}")
        user_consent = input("\nDo you want to proceed sending this info to a public LLM? (y/n): ")

        if user_consent.lower().strip() != 'y':
            print(f"Your most recent input has been cleared from memory. Please re-phrase it for me.")
            return False

    return True


if __name__ == '__main__':
    agent = Agent().agent

    while True:
        try:
            config = {"configurable": {"thread_id": 53}}
            user_input = input("USER>\n")

            if os.getenv('FULLY_LOCAL') == 0 and len(user_input) == 0 or not can_proceed_safely(user_input, ModelFactory.local_model):
                continue

            response = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

            logger.info(response)

            print(f'AGENT>\n{response['messages'][-1].content}\n')
        except KeyboardInterrupt:
            break


