from langchain import OpenAI,ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=apikey, model="text-davinci-003", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

memory = ConversationBufferMemory(return_message=True)

conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt)

user_message = "Tell me about the history of the Internet."
response = conversation(user_message)
print(response)

# User sends another message
user_message = "Who are some important figures in its development?"
response = conversation(user_message)
print(response)  # Chatbot responds with names of important figures, recalling the previous topic

