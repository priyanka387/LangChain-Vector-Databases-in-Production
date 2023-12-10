from langchain.output_parsers import PydanticOutputParser
from typing import List
from pydantic import BaseModel, Field, validator


class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")
    
    @validator('words')
    def not_start_with_number(cls, field):
      for item in field:
        if item[0].isnumeric():
          raise ValueError("The word can not start with numbers!")
      return field
    
    @validator('reasons')
    def end_with_dot(cls, field):
      for idx, item in enumerate( field ):
        if item[-1] != ".":
          field[idx] += "."
      return field
    
parser = PydanticOutputParser(pydantic_object=Suggestions)

from langchain.prompts import PromptTemplate

template = """
Offer a list of suggestions to substitute the specified target_word based on the presented context and the reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format_prompt(
			target_word="behaviour",
			context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

from langchain.llms import Cohere
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

#apikey = os.getenv("OPENAI_API_KEY")
apikey = os.getenv("COHERE_API_KEY")


model = Cohere(cohere_api_key = apikey, temperature=0)
#model = OpenAI(model = 'text-davinci-003',openai_api_key=apikey, temperature=0)
output = model(model_input.to_string())
parser.parse(output)
print(output)

