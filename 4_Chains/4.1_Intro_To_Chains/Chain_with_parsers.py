from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain import PromptTemplate, OpenAI, LLMChain
import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")
llm = OpenAI(openai_api_key = apikey, model="text-davinci-003", temperature=0)


output_parser = CommaSeparatedListOutputParser()
template = """List all possible words as substitute for 'artificial' as comma separated."""

llm_chain = LLMChain(
    llm=llm, 
    prompt=PromptTemplate(template=template, output_parser=output_parser, input_variables=[]),
    output_parser= output_parser
)

response = llm_chain.predict()
print(response)

        