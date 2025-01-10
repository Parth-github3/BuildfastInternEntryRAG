######## Requirements #######

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

########## Streamlit app ###########

# Title
st.title("Basic RAG App built on Gemini Model")

with st.sidebar:
    uploaded_files = st.file_uploader(
        "Choose a pdf file", accept_multiple_files=True, type="pdf"
    )
    
    def extract():
    
        extracted_text = []
        for file in uploaded_files:
            with PyPDFLoader.open(file) as pdf:
                for page in pdf.pages:
                    extracted_text.append(page.extract_text())
                   
        return extracted_text
    res= extract()

    # loader = PyPDFLoader(res)
    # datas = loader.load()




    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(res)
        


# Get file for RAG (Only pdf)
# loader = PyPDFLoader("Parth kundlini.pdf")
# data = loader.load()

# Processing the data in files
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# docs = text_splitter.split_documents(datas)
# docs = ' '.join([str(s) for s in docs])
# Creating and storing the embeddings of data in Chroma Vectorstore
#vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

#vectorstore = FAISS.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# from langchain_chroma import Chroma

# vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
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
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initializing the llm
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0,max_tokens=None,timeout=None)

# Getting the queries of user
query = st.chat_input("Say something: ") 
# prompt = query

# Defining the system prompt
# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer the question.  "
#     "If you don't know the answer, say that you "
#     "don't know. Use minimum of three sentences and keep the "
#     "answer concise."
#     "\n\n"
#     "{context}"
# )

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_TEMPLATE,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

from langchain_core.messages import HumanMessage


# Taking the action for provided query
if query:
    
    # question_answer_chain = create_stuff_documents_chain(llm, prompt)
    response= document_chain.invoke(
    {
        "context": docs,
        "messages": [
            HumanMessage(content=query)
        ],
    }
)
    # rag_chain = create_retrieval_chain(retriever, document_chain)

    # response = document_chain.invoke({"input": query})
    #print(response["answer"])

    st.write(response)