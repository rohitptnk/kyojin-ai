from fastapi import FastAPI, Request, UploadFile, File
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    docs = []
    for file in files:
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    return {"status": "success", "num_files": len(files), "num_docs": len(docs)}