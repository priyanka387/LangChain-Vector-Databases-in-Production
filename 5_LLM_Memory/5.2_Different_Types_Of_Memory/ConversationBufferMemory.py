from langchain import OpenAI, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=apikey, model="text-davinci-003", temperature=0)

conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=ConversationBufferMemory()
)
conversation.predict(input="Hello!")


template = """You are a customer support chatbot for a highly advanced customer support AI 
for an online store called "Galactic Emporium," which specializes in selling unique,
otherworldly items sourced from across the universe. You are equipped with an extensive
knowledge of the store's inventory and possess a deep understanding of interstellar cultures. 
As you interact with customers, you help them with their inquiries about these extraordinary
products, while also sharing fascinating stories and facts about the cosmos they come from.

{chat_history}
Customer: {customer_input}
Support Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "customer_input"], 
    template=template
)
chat_history=""

convo_buffer = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)
print(conversation.prompt.template)



convo_buffer("I'm interested in buying items from your store")
convo_buffer("I want toys for my pet, do you have those?")
convo_buffer("I'm interested in price of a chew toys, please")

import tiktoken

def count_tokens(text: str) -> int:
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = tokenizer.encode(text)
    return len(tokens)

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
]

total_tokens = 0
for message in conversation:
    total_tokens += count_tokens(message["content"])

print(f"Total tokens in the conversation: {total_tokens}")
