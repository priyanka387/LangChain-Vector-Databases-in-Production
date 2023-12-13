from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import Cohere
from langchain.document_loaders import SeleniumURLLoader
from langchain import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("Cohere_API_KEY")
deeplake_token = os.getenv("ACTIVELOOP_TOKEN")
os.environ["ACTIVELOOP_TOKEN"] = deeplake_token

# we'll use information from the following articles
urls = ['https://en.wikipedia.org/wiki/Large_language_model',
        'https://en.wikipedia.org/wiki/LangChain',
        'https://en.wikipedia.org/wiki/Prompt_engineering',
        'https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)',
        'https://jalammar.github.io/illustrated-bert/',
        'https://jalammar.github.io/illustrated-transformer/',
        'https://docs.cohere.com/docs/prompt-engineering'
        ]

# use the selenium scraper to load the documents
loader = SeleniumURLLoader(urls=urls)
docs_not_splitted = loader.load()

# we split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs_not_splitted)

# create embeddings
embeddings = CohereEmbeddings(
    cohere_api_key=apikey,
    model="embed-english-light-v2.0"
)

# create Deep Lake dataset
my_activeloop_org_id = "priyankapathak222"
my_activeloop_dataset_name = "Elearning_Bot"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# add documents to our Deep Lake dataset
db.add_documents(docs)

# let's see the top relevant documents to a specific query
#query = "what is langchain and what is it's use?"
#docs = db.similarity_search(query)
#print(docs[0].page_content)

#Create a template and use it in the prompt for the bot
template = """" You are an advanced AI Chatbot that gently answer the questions
you know the following context information.
{chunks_formatted}
Answer to the following question from a customer. Use only information from the previous context information. Do not invent stuff.
Question : {query}
Answer : """

prompt = PromptTemplate(
    input_variables=["chunks_formatted","query"],
    template = template
)

##User Question
query = "How can we do prompt engineering?"

#retreive the relevant chunk
docs = db.similarity_search(query)
retrieved_chunk = [doc.page_content for doc in docs]
#print(retrieved_chunk)

#format the prompt
chunks_formatted = "\n\n".join(retrieved_chunk)
prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)

#generate the answer
llm = Cohere(cohere_api_key = apikey, temperature=0)
answer = llm(prompt_formatted)
print(answer)





