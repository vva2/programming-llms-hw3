import os
from pprint import pprint

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages
from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough


load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


if __name__ == '__main__':
    trimmer = trim_messages(
        max_tokens=40,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability. You should reply in {language}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = (
            RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
            | prompt
            | model
    )

    with_message_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")

    config = {"configurable": {"session_id": "abc2"}}

    messages = [
        # SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]

    response = with_message_history.invoke(
        {
            "messages": messages + [
                HumanMessage(content="what math problem did i ask?")
            ],
            "language": "english"
        },
        config=config
    )

    pprint(response.content)


