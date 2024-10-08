import os
import re
import requests
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver  # an in-memory checkpointer
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_text_splitters import CharacterTextSplitter
from langchain_milvus import Zilliz
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()
model = "llama-3.1-70b-versatile"

DIRECTORY_PATH = "demo_notes/"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 40

loader = PyPDFDirectoryLoader(DIRECTORY_PATH)
pdf_content = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                      chunk_overlap=CHUNK_OVERLAP)
docs = text_splitter.split_documents(pdf_content)

embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5",
                             inference_mode='local')

@st.cache_resource(ttl="1h")
def configure_retriever():
    qdrant = Qdrant.from_documents(docs,
                                embeddings,
                                path="deploy/calculus",
                                collection_name="course_documents",
                                force_recreate=True)

    retriever = qdrant.as_retriever(k=4)
    return retriever



vectorstore = Zilliz(
    embedding_function=embeddings,
    collection_name="LangChainCollection",
    connection_args={
        "uri":st.secrets["ZILLIZ_URI"],
        "token":st.secrets["ZILLIZ_TOKEN"],
    }
)

retriever = vectorstore.as_retriever(k=4)

# retriever = configure_retriever()

@tool
def wolfram_call(query: str) -> str:
    """Calls the wolfram alpha api on query.

    Args:
        query (str): The query to use as a parameter in the function call. For example, "Solve 3x=5"

    Returns:
        str: Returns the result of the api call on the query.
    """
    params = {
        "input": query,
        "appid": st.secrets["WOLFRAM_ALPHA_APPID"],
        "format": "plaintext",
        "output": "json",
    }
    response = requests.get("https://api.wolframalpha.com/v2/query",
                            params=params)
    full_response = response.json()
    pods = [x["subpods"] for x in full_response["queryresult"]["pods"]]
    return str(pods)


@tool
def notes_search(query: str) -> str:
    """Calls the vector database of calculus notes to get any definitions, theorems, axioms, proofs, information, examples.
    The database contain notes for Calculus 1, 2 and 3. It covers the following topics:
    limits, derivatives, derivative applications, integrals, integral applications, integration techniques, more integral applications, parametric and polar, series and sequences, vectors, 3-d space, partial derivatives, applications of partial derivatives, multiple integrals, line integrals, surface integrals, various calculus proofs, review of trignometry and functions.

    Args:
        query (str): the query sent to the vector database.

    Returns:
        str: Returns the documents retrieved from the vector database based on the query.
    """
    return retriever.invoke(query)


tools = [wolfram_call, notes_search]
llm = ChatGroq(temperature=0, model=model)
def get_agent():

    system = """You are a mathematical assistant for a course on calculus. 
    Create a plan to solve the question and then solve it in a step-by-step manner.
    You have 2 tools available. Before using any tool, explain the logic behind using it and the arguments given to the tool.
    Use the wolfram tool wherever it is possible to create a valid query to solve any algebra required in the steps of the problem.
    Use the notes_search tool as a reference whenever required or if you are unsure about anything to search the course notes for the precise definitions, theorems, axioms, proofs, information and examples in calculus 1,2 and 3 and related topics. Rewrite anything retrieved in your own words."""

    agent_memory = MemorySaver()
    agent = create_react_agent(llm, tools, state_modifier=system, checkpointer=agent_memory)
    return agent


# Setup agent and QA chain
agent = get_agent()
config = {"configurable": {"thread_id": "test-thread"}}

st.title("WizLearnrAI")


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, HumanMessage):
            st.write("### Human Message")
            st.write(message.content)
        elif isinstance(message, AIMessage):
            st.write("## AI Message")
            if message.content:
                st.write(message.content)
            if "tool_calls" in message.additional_kwargs:
                st.write(message.additional_kwargs["tool_calls"])
        elif isinstance(message, ToolMessage):
            st.write("### Tool Response")
            st.write(message.content)
        else:
            st.write("### Error")

user_query = st.text_input("Enter your query:")
inputs = {"messages": [("user", user_query)]}
if st.button("Submit"):
        print_stream(agent.stream(inputs, stream_mode="values", config=config))
        

