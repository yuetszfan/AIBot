#import model for itinerary plan bot
from langchain.chat_models import ChatVertexAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
import re

#set parameters for model
chat = ChatVertexAI(max_output_tokens=1024,
    temperature=0,
    top_p=0.8,
    top_k=40,
    model_name='chat-bison@001',                
    verbose=True)

#scenario 1: no Tokyo Tower
messages = [
    SystemMessage(content="You are a helpful assistant that figure out travel plans based on user's request."),
    HumanMessage(content="give me 4-day trip itinerary to Tokyo. Tokyo Tower must not appear in the plan. Electronics store is a must"),
]
first_response = str(chat(messages)) #transform data type so \n could be replaced
first_resp = re.sub(r'\\n', '\n', first_response)
print(first_resp)

#scenario 2: add Tokyo Tower
messages = [
    SystemMessage(content="You are a helpful assistant that adjust the above travel plan."),
    HumanMessage(content="Adjust the 4-day plan and add Tokyo Tower. Show the adjusted plan."),
]
second_response = str(chat(messages)) #transform data type so \n could be replaced
second_resp = re.sub(r'\\n', '\n', second_response)
print(second_resp)

#additional tool:search for tourist spot open hour
from langchain.utilities import SerpAPIWrapper
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms.vertexai import VertexAI
import os
os.environ["SERPAPI_API_KEY"] = "6b195226153ea5dd991a03b2e3f4f008b5ec608b1c271429716eb649b0b29832"

llm = VertexAI(max_output_tokens=1024, verbose=True)
search = SerpAPIWrapper()
tools = [
    Tool(
        verbose=True,
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]
agent_executor = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)
result = agent_executor.run("give me open hour for Sensō-ji Temple")
print(result)

#import modules for insurance bot
import glob
import pprint
from typing import Any, Iterator, List

from langchain.agents import AgentType, initialize_agent
from langchain.document_loaders import TextLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.tools import tool
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from tqdm import tqdm

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=1024,
    temperature=0,
    top_p=0.8,
    top_k=40,
)

embedding = VertexAIEmbeddings()

def chunks(lst: List[Any], n: int) -> Iterator[List[Any]]:
    """Yield successive n-sized chunks from lst.

    Args:
        lst: The list to be chunked.
        n: The size of each chunk.

    Yields:
        A list of the next n elements from lst.
    """

    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def load_docs_from_directory(dir_path: str) -> List[Document]:
    """Loads a series of docs from a directory.

    Args:
      dir_path: The path to the directory containing the docs.

    Returns:
      A list of the docs in the directory.
    """

    docs = []
    for file_path in glob.glob(dir_path):
        loader = TextLoader(file_path)
        docs = docs + loader.load()
    return docs

def create_retriever(top_k_results: int, dir_path: str) -> VectorStoreRetriever:
    """Create a recipe retriever from a list of top results and a list of web pages.

    Args:
        top_k_results: number of results to return when retrieving
        dir_path: List of web pages.

    Returns:
        A recipe retriever.
    """

    BATCH_SIZE_EMBEDDINGS = 5
    docs = load_docs_from_directory(dir_path=dir_path)
    doc_chunk = chunks(docs, BATCH_SIZE_EMBEDDINGS)
    for (index, chunk) in tqdm(enumerate(doc_chunk)):
        if index == 0:
            db = FAISS.from_documents(chunk, embedding)
        else:
            db.add_documents(chunk)

    retriever = db.as_retriever(search_kwargs={"k": top_k_results})
    return retriever

insurance_retriever = create_retriever(top_k_results=2, dir_path="./insurance/*")

@tool(return_direct=True)
def retrieve_insurances(query: str) -> str:
    """
    Searches the insurance catalog to find suitable insurances for the query.
    Return the output without processing further.
    """
    docs = insurance_retriever.get_relevant_documents(query)

    return (
        f"Here are my recommendations. Select the insurance you would like to explore further about {query}: [START CALLBACK FRONTEND] "
        + str([doc.metadata for doc in docs])
        + " [END CALLBACK FRONTEND]"
    )

@tool
def insurance_selector(path: str) -> str:
    """
    Use this when the user mentions 'Selecting'.
    """
    return "Great choice! I can show you details of this insurance. What aspect are you interested in?"

docs = load_docs_from_directory("./insurance/*")
insurance_detail = {doc.metadata["source"]: doc.page_content for doc in docs}

@tool
def get_insurance_detail(path: str) -> str:
    """
    Use this when the user requests details about the specific insurance. For example 'Show me coverage scope of this insurance'
    Return the specific details the user requests,and then ask 'Would you like me to help you buy this insurance?'
    """
   
    try:
        return insurance_detail[path]
    except KeyError:
        return "Could not find the details for this insurance"

@tool
def order_insurance(str):
    """
    Use this when the user says 'Yes, order this insurance for me'
    Return 'OK,I've ordered the insurance. Safe flight and enjoy your trip!'
    """

memory = ConversationBufferMemory(memory_key="chat_history")
memory.clear()

tools = [
    retrieve_insurances,
    insurance_selector,
    get_insurance_detail,
    order_insurance
]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)

agent.run("I'm going on a 3-day trip to Japan. Can you recommend suitable travel insurance?")
agent.run("Selecting ./insurance/NTA.txt")
agent.run("Explain what Accidental Death Benefit is")
agent.run("Yes,order this insurance for me")
agent.run("Thank you, that's everything!")
