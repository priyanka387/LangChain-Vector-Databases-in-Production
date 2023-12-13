text = """
Objection: "There's no money."
It could be that your prospect's business simply isn't big enough or generating enough cash right now to afford a product like yours. Track their growth and see how you can help your prospect get to a place where your offering would fit into their business.

Objection: "We don't have any budget left this year."
A variation of the "no money" objection, what your prospect's telling you here is that they're having cash flow issues. But if there's a pressing problem, it needs to get solved eventually. Either help your prospect secure a budget from executives to buy now or arrange a follow-up call for when they expect funding to return.

Objection: "We need to use that budget somewhere else."
Prospects sometimes try to earmark resources for other uses. It's your job to make your product/service a priority that deserves budget allocation now. Share case studies of similar companies that have saved money, increased efficiency, or had a massive ROI with you.
"""

# Split the text into a list using the keyword "Objection: "
objections_list = text.split("Objection: ")[1:]  # We ignore the first split as it is empty

# Now, prepend "Objection: " to each item as splitting removed it
objections_list = ["Objection: " + objection for objection in objections_list]

import os
import re
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

class DeepLakeLoader:
    def __init__(self, source_data_path):
    self.source_data_path = source_data_path
    self.file_name = os.path.basename(source_data_path) # What we'll name our database 
    self.data = self.split_data()
    if self.check_if_db_exists():
        self.db = self.load_db()
    else:
        self.db = self.create_db()
        
        def split_data(self):  
    """  
    Preprocess the data by splitting it into passages.  

    If using a different data source, this function will need to be modified.  

    Returns:  
        split_data (list): List of passages.  
    """  
    with open(self.source_data_path, 'r') as f:  
        content = f.read()  
    split_data = re.split(r'(?=\d+\. )', content)
    if split_data[0] == '':  
        split_data.pop(0)  
    split_data = [entry for entry in split_data if len(entry) >= 30]  
    return split_data


def load_db(self):
    """  
    Load the database if it already exists.  

    Returns:  
        DeepLake: DeepLake object.  
    """  
    return DeepLake(dataset_path=f'deeplake/{self.file_name}', embedding_function=OpenAIEmbeddings(), read_only=True)  

def create_db(self):  
    """  
    Create the database if it does not already exist.  

    Databases are stored in the deeplake directory.  

    Returns:  
        DeepLake: DeepLake object.  
    """  
    return DeepLake.from_texts(self.data, OpenAIEmbeddings(), dataset_path=f'deeplake/{self.file_name}')


def query_db(self, query):  
    """  
    Query the database for passages that are similar to the query.  

    Args:  
        query (str): Query string.  

    Returns:  
        content (list): List of passages that are similar to the query.  
    """  
    results = self.db.similarity_search(query, k=3)  
    content = []  
    for result in results:  
        content.append(result.page_content)  
    return content

db = DeepLakeLoader('data/salestesting.txt')

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

chat = ChatOpenAI()
system_message = SystemMessage(content=objection_prompt)
human_message = HumanMessage(content=f'Customer objection: {detected_objection} | Relevant guidelines: {results}')

response = chat([system_message, human_message])






