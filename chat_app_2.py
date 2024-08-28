import os
import tempfile
import requests
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver  # an in-memory checkpointer
from langgraph.prebuilt import create_react_agent
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
model = "llama-3.1-70b-versatile"

st.set_page_config(page_title="Agentic Wizlearnr")
st.title("Calculus Assistant")

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())
        
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create embeddings and store in vectordb
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5",
                             inference_mode='local')
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    
    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k":2, "fetch_k":4})
    
    return retriever


uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    # st.stop()

retriever = configure_retriever(uploaded_files)

# Setup tools
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
        "appid": os.environ.get("WOLFRAM_ALPHA_APPID"),
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

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()

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

result = agent.invoke(
    {
        "messages": [
            ("user", "Solve 2+3")
        ]
    },
    config,
)["messages"][-1].content

# if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
#     msgs.clear()
#     msgs.add_ai_message("How can I help you?")

# avatars = {"human": "user", "ai": "assistant"}
# for msg in msgs.messages:
#     st.chat_message(avatars[msg.type]).write(msg.content)

# if user_query := st.chat_input(placeholder="Ask me anything!"):
#     st.chat_message("user").write(user_query)

#     with st.chat_message("assistant"):
#         # retrieval_handler = PrintRetrievalHandler(st.container())
#         # stream_handler = StreamHandler(st.empty())
#         #response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
#         response = agent.invoke(
            # {
            #     "messages": [
            #         ("user", user_query)
            #     ]
            # },
            # config,
#         )