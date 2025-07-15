# This code is part of a LangChain application that uses Groq's LLM for conversational retrieval from documents.
# This connects to the docchat page.

import os
from typing import List
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# print(GROQ_API_KEY)

# -- Load Documents --
ALL_DOCS = []
async def load_docs(files: List[UploadFile]):
    global ALL_DOCS
    os.makedirs("tmp", exist_ok=True)
    ALL_DOCS = []  # Reset on each upload
    for file in files:
        file_path = os.path.join("tmp", file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        ALL_DOCS.extend(docs)
    return {"status": "success", "num_files": len(files), "num_docs": len(ALL_DOCS)}

def get_loaded_docs():
    return ALL_DOCS

# docs[21].metadata


def bot(docs, user_message):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )

    # -- Create Embeddings and Vector Store --
    import getpass
    import os

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # -- Add Chat Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # -- Create Conversational Retrieval Chain
    # Using Groq's LLM for the chat model
    chat = ChatGroq(
        temperature=0.0, 
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=False
    )

    result = qa.invoke(user_message)
    print(result['answer'])

    return result['answer']


# Example usage
# loader = PyPDFLoader("Choi_2020_J._Cosmol._Astropart._Phys._2020_045.pdf")
# docs = loader.load()
# print(bot(docs, "What is this doc about?"))
