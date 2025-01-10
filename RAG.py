######## Requirements #######

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

########## Streamlit app ###########

# Title
st.title("Basic RAG App built on Gemini Model")

# Get file for RAG (Only pdf)
loader = PyPDFLoader("Parth kundlini.pdf")
data = loader.load()

#Processing the data in files
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(data)

# Creating and storing the embeddings of data in FAISS Vectorstore
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

index = faiss.IndexFlatL2(len(embeddings.embed_query("Instantiation Query")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

from uuid import uuid4
uuids = [str(uuid4()) for _ in range(len(docs))]

vector_store.add_documents(documents=docs, ids=uuids)

# Creating the retriever object to retrieve the data directly from the Vectorstore
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10})

# Initializing the llm
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0,max_tokens=None,timeout=None)

# Getting the queries of user
query = st.chat_input("Say something: ") 

#Creating the Templates and prompts
SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context and make sure to provide a concise answer. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""

# QA prompt
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_TEMPLATE,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

#Creating the chain
document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

# Taking the action for provided query and getting response
if query:
    st.markdown("## Input")
    st.write(query)

    response= document_chain.invoke(
    {
        "context": docs,
        "messages": [
            HumanMessage(content=query)
        ],
    }
)
    st.markdown("## Output")
    st.write(response)