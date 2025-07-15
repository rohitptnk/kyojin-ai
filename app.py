from fastapi import FastAPI, Request, UploadFile, File, Form
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from rag import load_docs, get_loaded_docs, bot

app = FastAPI()

# Mount static folder for CSS, JS, images, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/docchat", response_class=HTMLResponse)
async def read_docchat(request: Request):
    return templates.TemplateResponse("docchat.html", {"request": request})

@app.post("/upload/")
async def upload(files: List[UploadFile] = File(...)):
    return await load_docs(files)

@app.post("/query/")
async def query(question: str = Form(...)):
    docs = get_loaded_docs()
    if not docs:
        return {"error": "No documents uploaded. Upload a document to get started."}
    # Here you’d do embedding + retrieval + answer
    response = bot(docs, question)
    if not response:
        return {"error": "No reply from bot"}
    return {"reply": response}
