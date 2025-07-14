# This code is part of a LangChain application that uses Groq's LLM for conversational retrieval from documents.
# This connects to the docchat page.

# -- Load Documents --
import os
from typing import List
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader

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

# # -- Split into chunks --
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1500,
#     chunk_overlap = 150
# )

# splits = text_splitter.split_documents(docs)
# # len(splits)

# # splits[2].page_content

# from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# # -- Embed the chunks and store in vectorstore
# from langchain.vectorstores import FAISS
# vectorstore = FAISS.from_documents(docs, embeddings)

# # -- Add Chat Memory
# from langchain.memory import ConversationBufferMemory
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# from langchain_groq import ChatGroq
# import os
# chat = ChatGroq(temperature=0.0, model="llama-3.3-70b-versatile")
# # chat

# # response = chat.invoke("What is the capital of India?")
# # print(response.content)

# from langchain.chains import ConversationalRetrievalChain
# qa = ConversationalRetrievalChain.from_llm(
#     llm=chat,
#     retriever=vectorstore.as_retriever(),
#     memory=memory,
#     verbose=True
# )

# # question = "Which Deep Learning is being used to remove the foreground?"
# # result = qa({"question": question})

# # result['answer']

# # question = "Is that better than CNN?"
# # result = qa({"question": question})

# # result['answer']

# # === Chat loop ===
# while True:
#     query = input("You: ")
#     if query.lower() in ["exit", "quit"]:
#         break
#     response = qa.run(query)
#     print("Bot:", response)

