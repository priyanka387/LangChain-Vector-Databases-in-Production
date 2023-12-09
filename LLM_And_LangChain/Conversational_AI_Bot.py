from langchain.llms import Cohere
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()
apikey = os.getenv("COHERE_API_KEY")
llm = Cohere(cohere_api_key = apikey, temperature=0)

# Define the input text
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))

'''# Determine the maximum number of tokens from documentation
max_tokens = 4097

# # we split the documents into smaller chunks

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
docs = text_splitter.split_documents(docs_not_splitted)

#process each chunk seperately
results = []
for chunk in text_chunks:
    result = llm.process(chunk)
    results.append(result)

#combine the results as needed
final_result = combine_results(results)'''