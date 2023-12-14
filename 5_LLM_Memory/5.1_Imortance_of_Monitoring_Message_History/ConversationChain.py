from langchain import OpenAI, ConversationChain

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=apikey, model="text-davinci-003", temperature=0)

conversation = ConversationChain(llm=llm, verbose=True)
output = conversation.predict(input="Hi there!")
print(output)

output = conversation.predict(input="In what scenarios extra memory should be used?")
output = conversation.predict(input="There are various types of memory in Langchain. When to use which type?")
output = conversation.predict(input="Do you remember what was our first message?")

print(output)