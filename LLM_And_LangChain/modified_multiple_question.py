from langchain import PromptTemplate
from langchain import HuggingFaceHub, LLMChain

import os
from dotenv import load_dotenv

load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")


multi_template = """Answer the following questions one at a time.
Questions:{questions}
Answers:
"""
long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])


qs_str = (
    "What is the capital city of France?\n" +
    "What is the largest mammal on Earth?\n" +
    "Which gas is most abundant in Earth's atmosphere?\n" +
		"What color is a ripe banana?\n"
)


# initialize Hub LLM
hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-large',
    model_kwargs={'temperature':0}
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=long_prompt,
    llm=hub_llm
)

# ask the user question about the capital of France
res = llm_chain.run(qs_str)
print(res)