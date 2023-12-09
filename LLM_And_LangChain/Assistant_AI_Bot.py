from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)


import os
from dotenv import load_dotenv
load_dotenv()


apikey = os.getenv("OPENAI_API_KEY")

#apikey = "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"

chat = ChatOpenAI(openai_api_key = apikey, model ="gpt-3.5-turbo", temperature=0)

template = "Yor an assistant that helps the users find information about movies"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "Find the information about the movie {movie_title}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

response = chat(chat_prompt.format_prompt(movie_title="Inception").to_messages())

print(response.content)