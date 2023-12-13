import langchain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Cohere

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("COHERE_API_KEY")

llm = Cohere(
    cohere_api_key=apikey
    # model="large"
)


prompt = PromptTemplate(
    input_variables = ["product"],
    template = "What is the good name of the company that makes {product}?"
)

chain = LLMChain(llm = llm, prompt=prompt)

#Run the chain only specifying the input variable
print(chain.run("eco friendly water bottle"))



