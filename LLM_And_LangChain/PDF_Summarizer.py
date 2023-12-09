from langchain.llms import Cohere
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import DeepLake
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
import gradio as gr

import os
from dotenv import load_dotenv
load_dotenv()
apikey = os.getenv("Cohere_API_KEY")
llm = Cohere(cohere_api_key=apikey, model = "large", temperature=0, max_token = 500)


def summarize_pdf(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)   
    return summary

input_pdf_path = gr.components.Textbox(label="Provide the PDF file path")
output_summary = gr.components.Textbox(label="Summary")

interface = gr.Interface(
    fn=summarize_pdf,
    inputs=input_pdf_path,
    outputs=output_summary,
    title="PDF Summarizer",
    description="Provide PDF file path to get the summary.",
).launch(share=True)