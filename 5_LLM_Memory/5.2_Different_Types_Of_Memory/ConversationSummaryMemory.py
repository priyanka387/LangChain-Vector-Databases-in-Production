from langchain.chains import ConversationChain
from langchain import OpenAI
from langchain.memory import ConversationSummaryMemory

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=apikey, model_name="text-davinci-003", temperature=0)


# Create a ConversationChain with ConversationSummaryMemory
conversation_with_summary = ConversationChain(
    llm=llm, 
    memory=ConversationSummaryMemory(llm=llm),
    verbose=True
)

# Example conversation
response = conversation_with_summary.predict(input="Hi, what's up?")
print(response)