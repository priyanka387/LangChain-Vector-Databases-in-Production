from langchain import PromptTemplate, OpenAI, LLMChain

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

template = "What is a word to replace the following: {word}?"

llm = OpenAI(openai_api_key = apikey, model="text-davinci-003", temperature=0)

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(template)
)

#we can directly pass the input to llm chain
# response = llm_chain("artificial")
# print(response)

#t is also possible to use the .apply() method to pass multiple inputs at once and receive a list for each input.
input_list = [
    {"word": "artificial"},
    {"word": "intelligence"},
    {"word": "robot"}
]

# llm_chain.apply(input_list)

#The .generate() method will return an instance of LLMResult, which provides more information. 
response = llm_chain.generate(input_list)
print(response)

