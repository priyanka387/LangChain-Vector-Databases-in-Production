from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("D:/My_Resume/Priyanka-Kumari.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20,
    length_function = len
)

docs = text_splitter.split_documents(pages)
for doc in docs:
    print(doc)
    
print(len(docs))
