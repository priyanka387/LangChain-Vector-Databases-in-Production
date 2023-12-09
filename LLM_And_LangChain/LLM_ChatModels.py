from langchain.chat_models import ChatOpenAI
from langchain.schema import (HumanMessage, SystemMessage)

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(openai_api_key = apikey, model="gpt-3.5-turbo", temperature=0)


#messages = [
    #SystemMessage(content = "You are a helpful assistant that translates English to French"),
    #HumanMessage(content = "Trannslate the following senetnce: I love Programming")
#]

#response = chat(messages)
#print(response)"""

batch_messages = [
  [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate the following sentence: I love programming.")
  ],
  [
    SystemMessage(content="You are a helpful assistant that translates French to English."),
    HumanMessage(content="Translate the following sentence: J'aime la programmation.")
  ],
]
print( chat.generate(batch_messages) )


