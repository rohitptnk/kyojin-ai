# Imports
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
from langchain.chains import ConversationChain

# Select Chat Model
import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = init_chat_model("gpt-4o-mini", model_provider="openai")


# Select Embedding Model
import getpass
import os
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# Select Vector Store
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)


# Index Documents
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict

ALL_DOCS = [] # Load documents from the user
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

    
# Bot Function
def bot(docs, user_message):
    from typing import List
    from typing_extensions import Annotated, TypedDict

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)

    ## Desired schema for response
    class AnswerWithSources(TypedDict):
        """An answer to the question, with sources."""

        answer: str
        sources: Annotated[
            List[str],
            ...,
            "List of sources (author + year) used to answer the question",
        ]

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: AnswerWithSources

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        structured_llm = llm.with_structured_output(AnswerWithSources)
        response = structured_llm.invoke(messages)
        return {"answer": response}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    # Get answer and main source
    import json

    result = graph.invoke({"question": "What is Chain of Thought?"})
    print(json.dumps(result["answer"], indent=2))

    # Get context
    print(result["context"])


# Example usage
if __name__=="__main__":
    loader = PyPDFLoader("Choi_2020_J._Cosmol._Astropart._Phys._2020_045.pdf")
    docs = loader.load()
    print(bot(docs, "What is this doc about?"))