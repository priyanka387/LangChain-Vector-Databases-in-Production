from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate


import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

#Create our example
examples = [
    {"query" : "what's the weather like?",
     "answer" : "It's raining cats and dogs, better bring an umbrella"
     },
     {"query" : "How old are you?",
      "answer" : "Age is just a number, but I'm timeless"
      }
]

#Create an example template
example_template = """
User : {query}
AI : {answer} """

#Create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables = ["query", "answer"],
    template = example_template
)

#now break our previous prompt into prefix and suffix
# the prefix is our instructions

prefix = """The following are excerpts from conversation with an AI
assistant. The assisatant is known for its humor and wit, providing entertaining and amusing 
responses to user's question. Here are some examples:"""

#and the suffix our user input and output indicator
suffix = """
user:{query}
Ai = """

#now create few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    #example_separator="\n\n"
)

chat = ChatOpenAI(openai_api_key= apikey,  model = "gpt-3.5-turbo", temperature=0)

chain = LLMChain(llm=chat, prompt=few_shot_prompt_template)
response = chain.run("You are beutiful")
print(response)
