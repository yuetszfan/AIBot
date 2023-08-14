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
