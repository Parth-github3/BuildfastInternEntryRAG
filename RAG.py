######## Requirements #######

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

########## Streamlit app ###########

# Title
st.title("Basic RAG App built on Gemini Model")

# Get file for RAG (Only pdf)
uploaded_files = st.file_uploader(
    "Choose a PDF file", accept_multiple_files=False
)
loader = PyPDFLoader("uploaded_files")
data = loader.load()

# Processing the data in files
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(data)

# Creating and storing the embeddings of data in Chroma Vectorstore
vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# Creating the retriever object to retrieve the data directly from the Vectorstore
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initializing the llm
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0,max_tokens=None,timeout=None)

# Getting the queries of user
query = st.chat_input("Say something: ") 
prompt = query

# Defining the system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question.  "
    "If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Taking the action for provided query
if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    #print(response["answer"])

    st.write(response["answer"])