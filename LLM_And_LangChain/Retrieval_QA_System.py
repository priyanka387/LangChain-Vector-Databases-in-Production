from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Cohere
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("COHERE_API_KEY")
deeplake_token = os.getenv("ACTIVELOOP_TOKEN")
os.environ["ACTIVELOOP_TOKEN"] = deeplake_token

#Initiate the LLM and embeddings models
llm = Cohere(cohere_api_key = apikey, temperature=0)
embeddings = CohereEmbeddings(model="embed-english-light-v2.0")

#Create our document
texts = [
    "Nepolean Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638"
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,chunk_overlap = 0)
docs = text_splitter.create_documents(texts)


#Create DeepLake Dataset
my_activeloop_org_id = "priyankapathak222" 
my_activeloop_dataset_name = "text_embeddings1"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)


# add documents to our Deep Lake dataset
db.add_documents(docs)

#let's create a RetrievalQA chain:
retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm,
	chain_type="stuff",
	retriever=db.as_retriever()
)

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

tools = [Tool(
    name = "Retrieval QA System",
    func = retrieval_qa.run,
    description= "Useful for answer questions"
)]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)

response = agent.run("When was Nepolean born")
print(response)


""""
#load the existing deep lake dataset and specify the embeddings
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

#create new documents
texts = [
    "Lady Gaga was born in 28 1986",
    "Michael Jeffery Jordan was born in 17 february 1963"
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

#add document to our Deep Lake dataset
db.add_documents(docs)

#We then recreate our previous agent and ask a question that can be answered only by the last documents added.
#Initiate the wrapper class 
llm = Cohere(cohere_api_key=apikey, temperature=0)

#Create the retrievar from db
tools = [
    TOOL(
        name = "Retieval QA System",
        func = retrieval_qa.run,
        description = "useful for answering questions"
    )
]

#create an agent that uses tool
agent = initialize_agent(
    tools,
    llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTIION, verbose = True)

response = agent.run("When was Michael Jordan born")
print(response)"""








                                            