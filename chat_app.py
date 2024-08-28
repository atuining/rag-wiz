import os
import re
import requests
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langgraph.checkpoint.memory import MemorySaver  # an in-memory checkpointer
from langgraph.prebuilt import create_react_agent
from IPython.display import Latex
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

def print_stream(stream, is_new):
    
    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None
    
    while msg := next(stream, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()
            
            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        
        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)
                
                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                
                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                    f"""Tool Call: {tool_call["name"]}""",
                                    state="running" if is_new else "complete",
                                )
                            call_results[tool_call["id"]] = status
                            status.write("Input:")
                            status.write(tool_call["args"])

                        # Expect one ToolMessage for each tool call.
                        for _ in range(len(call_results)):
                            tool_result = next(stream)
                            if not tool_result.type == "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()
                            
                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            # In case of an unexpected message type, log an error and stop
            case _: 
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()

def old_print_stream(stream):
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
        old_print_stream(agent.stream(inputs, stream_mode="values", config=config))
        

