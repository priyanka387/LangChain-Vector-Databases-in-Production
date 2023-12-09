from langchain import PromptTemplate
from langchain import HuggingFaceHub, LLMChain

import os
from dotenv import load_dotenv

load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")


template = """
Question : {question}
Answer : """

prompt = PromptTemplate(
    template = template,
    input_variables = ['question']
)

question = "What is the capital city of France?"


# initialize Hub LLM
hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-large',
    model_kwargs={'temperature':0}
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# ask the user question about the capital of France
print(llm_chain.run(question))
