from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris.")
]

prompt = HumanMessage(
    content="I'd like to know more about the city you just mentioned."
)
# add to messages
messages.append(prompt)

llm = ChatOpenAI(openai_api_key = apikey, model ="gpt-3.5-turbo")

response = llm(messages)
print(response)