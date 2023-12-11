from langchain.llms import GooglePalm, Cohere, OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import DeepLake
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("GOOGLE_API_KEY")
deeplake_token = os.getenv("ACTIVELOOP_TOKEN")
os.environ["ACTIVELOOP_TOKEN"] = deeplake_token
apikey = os.getenv("COHERE_API_KEY")
#apikey = os.getenv("OPENAI_API_KEY")

text = """Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
Google is offering developers access to one of its most advanced AI language models: PaLM.
The search giant is launching an API for PaLM alongside a number of AI enterprise tools
it says will help businesses “generate text, images, code, videos, audio, and more from
simple natural language prompts.”

PaLM is a large language model, or LLM, similar to the GPT series created by OpenAI or
Meta’s LLaMA family of models. Google first announced PaLM in April 2022. Like other LLMs,
PaLM is a flexible system that can potentially carry out all sorts of text generation and
editing tasks. You could train PaLM to be a conversational chatbot like ChatGPT, for
example, or you could use it for tasks like summarizing text or even writing code.
(It’s similar to features Google also announced today for its Workspace apps like Google
Docs and Gmail.)
"""

# write text to local file
with open("my_file.txt", "w") as file:
    file.write(text)

# use TextLoader to load text from local file
loader = TextLoader("my_file.txt")
docs_from_file = loader.load()

#Create a text splitter
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)

#split the text into chunks
docs = text_splitter.split_documents(docs_from_file)
print(len(docs))


print(len(docs_from_file))

#Create embeddings
embeddings = CohereEmbeddings(
    cohere_api_key=apikey,
    model="embed-english-light-v2.0"
)

#Create deeplake dataset
my_activeloop_org_id = "priyankapathak222"
my_activeloop_dataset_name = "document_embeddings1"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

#add the document to our deep lake dataset
db.add_documents(docs)

#create the retriever from db
retriever = db.as_retriever()


#Create LLM
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

#Create Compressor
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
	base_compressor=compressor,
	base_retriever=retriever
)

#retrieving compressed documents
retrieved_docs = compression_retriever.get_relevant_documents(
    "How Google plans to challange OpenAI"
)

print(retrieved_docs[0].page_content)




