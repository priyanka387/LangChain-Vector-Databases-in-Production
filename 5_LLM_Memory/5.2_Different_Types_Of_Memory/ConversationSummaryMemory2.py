from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=apikey, model_name="gpt-3.5-turbo", temperature=0)


conversation_with_summary = ConversationChain(
    llm = llm,
    memory = ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40)
)

conversation_with_summary.predict(input="Hi, what's up?")
conversation_with_summary.predict(input="Just working on writing some documentation!")
response = conversation_with_summary.predict(input="For LangChain! Have you heard of it?")
print(response)



