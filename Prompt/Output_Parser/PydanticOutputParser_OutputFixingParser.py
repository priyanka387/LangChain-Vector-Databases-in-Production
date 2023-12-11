from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")

parser = PydanticOutputParser(pydantic_object=Suggestions)

#missformatted_output = '{"words": ["conduct", "manner"], "reasoning": ["refers to the way someone acts in a particular situation.", "refers to the way someone behaves in a particular situation."]}'

#parser.parse(missformatted_output)


from langchain.llms import OpenAI, Cohere
from langchain.output_parsers import OutputFixingParser

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("COHERE_API_KEY")
#apikey = os.getenv("OPENAI_API_KEY")

model = Cohere(cohere_api_key = apikey, temperature=0)
#model = OpenAI(model = 'text-davinci-003',openai_api_key=apikey, temperature=0)

missformatted_output = '{"words": ["conduct", "manner"]}'

outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)

outputfixing_parser.parse(missformatted_output)

outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
outputfixing_parser.parse(missformatted_output)