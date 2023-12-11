from langchain.output_parsers import CommaSeparatedListOutputParser
parser = CommaSeparatedListOutputParser()


from langchain.prompts import PromptTemplate

#Prepare the template
# Prepare the Prompt
template = """
Offer a list of suggestions to substitute the word '{target_word}' based the presented the following text: {context}.
{format_instructions}
"""


# Prepare the Prompt
prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format(
  target_word="behaviour",
  context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

#Loading the model
from langchain.llms import OpenAI, Cohere

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("COHERE_API_KEY")
#apikey = os.getenv("OPENAI_API_KEY")

model = Cohere(cohere_api_key = apikey, temperature=0)
#model = OpenAI(model = 'text-davinci-003',openai_api_key=apikey, temperature=0)

#Send the request to the model
output = model(model_input)
parser.parse(output)
print(output)
