from langchain import PromptTemplate, OpenAI, LLMChain

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")
llm = OpenAI(openai_api_key = apikey, model="text-davinci-003", temperature=0)

template = "Looking at the context of '{context}'. What is an appropriate word to replace the following: {word}?"

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=template, input_variables=["word", "context"]))

response = llm_chain.predict(word="fan", context="object")
print(response)
# or llm_chain.run(word="fan", context="object")

