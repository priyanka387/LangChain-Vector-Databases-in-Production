
from langchain.llms import Cohere
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import Cohere
from langchain.document_loaders import SeleniumURLLoader
from langchain import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("COHERE_API_KEY")
deeplake_token = os.getenv("ACTIVELOOP_TOKEN")
os.environ["ACTIVELOOP_TOKEN"] = deeplake_token


# # we'll use information from the following articles

urls = [
    "https://marg-darshan.com/", # margdarshan website 
    "https://jeemain.nta.nic.in/about-jeemain-2023/", # Govt NTA 
    "https://byjus.com/jee/jee-main/", # byjus
    "https://en.wikipedia.org/wiki/Joint_Entrance_Examination_%E2%80%93_Main", # wikipedia
    "https://engineering.careers360.com/exams/jee-main", # carrers360
    "https://rajneetpg2022.com/jee-main-exam-date/" # shiksha
]


# # use the selenium scraper to load the documents

loader = SeleniumURLLoader(urls=urls)
docs_not_splitted = loader.load()



# # we split the documents into smaller chunks

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
docs = text_splitter.split_documents(docs_not_splitted)


embeddings = CohereEmbeddings(
    cohere_api_key=apikey,
    model="embed-english-light-v2.0"
)



# # create Deep Lake dataset

my_activeloop_org_id = "abduljaweed"
my_activeloop_dataset_name = "margdarshan_qa_chatbot_v1"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# # add documents to our Deep Lake dataset
db.add_documents(docs)


# # let's see the top relevent documents to a specific query

# query = "What is margdarshan"
# docs = db.similarity_search(query)
# print(docs[0].page_content)





# let's write a prompt for a customer support chatbot that
# answer questions using information extracted from our db
template = """You are an exceptional customer support chatbot that gently answer questions.

You know the following context information.

{chunks_formatted}

Answer to the following question from a customer. Use only information from the previous context information. Do not invent stuff.

Question: {query}

Answer:"""

prompt = PromptTemplate(
    input_variables=["chunks_formatted", "query"],
    template=template
)


# the full pipeline

# user question

query = "What is margdarshan and what is the current JEE Main Update ?"

# retrieve relevent chunks

docs = db.similarity_search(query)

retrieved_chunks = [doc.page_content for doc in docs]

# format the prompt 

chunks_formatted = "\n\n".join(retrieved_chunks)
prompt_formatted = prompt.format(
    chunks_formatted=chunks_formatted,
    query=query
)

# generate answer 

llm = Cohere(
    cohere_api_key=apikey
    # model="large"
)
answer = llm(prompt_formatted)
print(answer)
