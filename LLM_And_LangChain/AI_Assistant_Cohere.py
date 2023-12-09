from langchain.llms import Cohere
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)


import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("COHERE_API_KEY")

llm = Cohere(cohere_api_key = apikey, model="large", temperature=0)

template = "Yor an assistant that helps the users find information about movies"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "Find the information about the movie {movie_title}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Create a chat prompt template from the messages
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Format the chat prompt with a specific movie title
formatted_chat_prompt = chat_prompt.format_prompt(movie_title="Inception")

# Get the response from the language model
response = llm(formatted_chat_prompt.to_messages())

# Print the content of the response
print(response.content)