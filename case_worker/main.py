import os
import subprocess
import nmap
# from zap_gpt import main as zap_gpt
import requests

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate

load_dotenv()

# TODO: Move to argparse
# TARGET =  "example.com"
MODEL = "ChatOpenAImini"

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "example.com", "Lov om folketrygd (folketrygdloven) - Lovdata.pdf")  # docs for RAG
db_dir = os.path.join(current_dir, "db")  # Chroma db directory
store_name = f"chroma_db_huggingface_example.com" # db name
persistent_directory = os.path.join(db_dir, store_name)  # db fullpath

# Split the document into chunks
loader = PyPDFLoader(file_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Display information about the split documents
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content}\n")

# Function to create and persist vector store
def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")

# 2. Hugging Face Transformers
print("\n--- Using Hugging Face Transformers ---")
embeddings = HuggingFaceEmbeddings(
    model_name="nlpaueb/legal-bert-small-uncased"
)
# embeddings = OpenAIEmbeddings()
create_vector_store(docs, embeddings, store_name)
print(f"Embedding demonstrations for {store_name}.")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10},
)

# Create a ChatOpenAI model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

# Contextualize question prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
qa_system_prompt = (
    "You are a cybersecurity professional specializing in finding vurnabilites. " 
    "Use your tools to explore and exploit the application the user wants to examine. "
    "Use the post request tool to interact with the application.  "
    "If you can't find any vurnabilites, just say that you "
    "cannot find any. Keep the answer concise. "
    "\n\n"
    "{context}"
)


# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain
    )


# Set Up ReAct Agent with Document Store Retriever (this can also just be defined as a template)
react_docstore_prompt = hub.pull("hwchase17/react")
# prompt = hub.pull("hwchase17/structured-chat-agent")

# Define rule query system
class QueryRuleSystem(BaseTool):
    name: str = "Query rule system"
    description: str = "Use this function to query a rule system."

    def _run(self, ip_address: str):
        """Run an nmap scan on a given IP address."""
        print(f"Running nmap scan on {ip_address}...")
        scanner = nmap.PortScanner()
        scan_result = scanner.scan(ip_address, arguments='-sC -sV -vv')
        return scan_result



tools = [
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", [])}
        ),
        description="useful for when you need to answer questions about the context",
    ),
    QueryRuleSystem(),
]

# Create the ReAct Agent with document store retriever
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_docstore_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True,
)

chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = agent_executor.invoke(
        {"input": query, "chat_history": chat_history})
    print(f"AI: {response['output']}")

    # Update history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))
