from langchain.llms import Cohere
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain.schema import OutputParserException

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("COHERE_API_KEY")
google_cse_id = os.environ.get("GOOGLE_CSE_ID")
google_api_key = os.environ.get("GOOGLE_API_KEY")

llm = Cohere(cohere_api_key=apikey, temperature=0)

prompt = PromptTemplate(
    input_variables=["query"],
    template="Write a summary of the following text: {query}"
)

summarize_chain = LLMChain(llm=llm, prompt=prompt)

# Next, we create the tools that our agent will use.
# Google search via api
search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for finding information about recent events"
    ),
    Tool(
        name='Summarizer',
        func=summarize_chain.run,
        description='useful for summarizing texts'
    )
]

# Create an agent to leverage the tool
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Perform the search and summarization
response = agent("What's the latest news about the Mars rover? Then please summarize the results.")

# Debugging: Print the raw output before parsing
print("Raw Output:", response)

# Attempt to parse the output
try:
    # Extracting relevant information from the string output
    summary_start_index = response.find("Could not parse LLM output:") + len("Could not parse LLM output:")
    summary_end_index = response.find("Would you like me to search for more information on any of the above rover missions?")

    # Extract the summary part from the response
    parsed_output = response[summary_start_index:summary_end_index].strip()

    print("Parsed Output:", parsed_output)
except OutputParserException as e:
    print("Parser Exception:", e)
    # Handle the exception or print additional information for debugging
