from langchain.document_loaders import ApifyDatasetLoader
from langchain.utilities import ApifyWrapper
from langchain.document_loaders.base import Document

import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv("APIFY_API_TOKEN")

apify = ApifyWrapper(apify_api_token=apikey)
loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://en.wikipedia.org/wiki/Large_language_model"}]},
    dataset_mapping_function=lambda dataset_item: Document(
        page_content=dataset_item["text"] if dataset_item["text"] else "No content available",
        metadata={
            "source": dataset_item["url"],
            "title": dataset_item["metadata"]["title"]
        }
    ),
)

docs = loader.load()
print(docs)

from langchain.text_splitter import RecursiveCharacterTextSplitter

#we split the documents into small chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=20, length_function=len
)

docs_split = text_splitter.split_documents(docs)
print(docs_split[0])

