import openai
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


import chromadb
chroma_client = chromadb.Client()

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")



# Function to upload PDF, split into chunks, and convert to embeddings
def process_pdf(pdf_path, model="gpt3.5-turbo"):
    # Load PDF document
    with open(pdf_path, "rb") as file:
        pdf_document = PyPDFLoader(pdf_path)
        

    
    # Split data to create chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)

    # Split document into chunks
    

    # Convert chunks to embeddings using OpenAIEmbeddings
    embeddings = []
    embeddings_model = OpenAIEmbeddings(model)
    for chunk in chunks:
        response = embeddings_model.embed(chunk)
        embeddings.append(response["embedding"])

    # Store embeddings in ChromaDB (replace with the appropriate storage mechanism)
    chromadb = chroma_client()
    for i, embedding in enumerate(embeddings):
        chromadb.store(f"chunk_{i}", embedding)

    return chunks, embeddings

# Function to summarize document using OpenAI language model
def summarize_document(document_chunks, model="gpt3.5-turbo"):
    concatenated_text = " ".join(document_chunks)
    response = openai.Completion.create(
        engine=model,
        prompt=concatenated_text,
        max_tokens=150
    )
    summary = response["choices"][0]["text"]
    return summary

# Example usage
pdf_path = "D:\My_Resume\Priyanka-Kumari.pdf"
chunks, embeddings = process_pdf(pdf_path)
document_summary = summarize_document(chunks)

# Print the document summary
print("Document Summary:")
print(document_summary)