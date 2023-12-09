from langchain.llms import Cohere
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import os
from dotenv import load_dotenv

load_dotenv()
apikey = os.getenv("COHERE_API_KEY")

llm = Cohere()
conversation = ConversationChain(llm = llm, verbose = True, memory = ConversationBufferMemory())

#Start the conversation
conversation.predict(input = "Tell me about yourself")

#Continue the conversation
conversation.predict(input="what can you do?")
conversation.predict(input = "How can you help me with data analysis")

#Display the conversation
print(conversation)

