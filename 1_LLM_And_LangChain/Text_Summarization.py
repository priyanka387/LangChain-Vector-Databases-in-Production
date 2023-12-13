from langchain.llms import Cohere
from langchain import PromptTemplate
from langchain.chains import LLMChain

import os
from dotenv import load_dotenv
load_dotenv()
apikey = os.getenv("COHERE_API_KEY")

summarization_template = "Summarize the following to one sentence:{text}"
summarization_prompt = PromptTemplate(input_variables = ["text"], template = summarization_template)

llm = Cohere.generate(self = Cohere , model='xlarge',prompts = str(summarization_prompt))

summarization_template = "Summarize the following to one sentence:{text}"
summarization_prompt = PromptTemplate(input_variables = ["text"], template = summarization_template)
summarization_chain = LLMChain(llm =llm, prompt=summarization_prompt)

text = "LangChain provides many modules that can be used to build language model applications. Modules can be combined to create more complex applications, or be used individually for simple applications. The most basic building block of LangChain is calling an LLM on some input. Let’s walk through a simple example of how to do this. For this purpose, let’s pretend we are building a service that generates a company name based on what the company makes."
summarized_text = summarization_chain.predict(text=text)
print(summarized_text)

