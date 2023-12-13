from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain import PromptTemplate, OpenAI, LLMChain

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")
llm = OpenAI(openai_api_key = apikey, model="text-davinci-003", temperature=0)

output_parser = CommaSeparatedListOutputParser()

conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())

response=conversation.predict(input="List all possible words as substitute for 'artificial' as comma separated.")
print(response)

response2= conversation.predict(input="And the next 4?")
print(response2)
