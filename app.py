from fastapi import FastAPI, Request, UploadFile, File, Form
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from rag import load_docs, get_loaded_docs, bot
from collections import defaultdict
from datetime import datetime

# Track usage per IP
query_counter = defaultdict(lambda: {"count": 0, "date": datetime.utcnow().date()})
upload_counter = defaultdict(lambda: {"count": 0, "date": datetime.utcnow().date()})

def increment_counter(counter, ip, limit):
    today = datetime.utcnow().date()
    if counter[ip]["date"] != today:
        counter[ip] = {"count": 0, "date": today}
    if counter[ip]["count"] < limit:
        counter[ip]["count"] += 1
    return counter[ip]["count"]


app = FastAPI()

# limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

@app.get("/about", response_class=HTMLResponse)
async def read_about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.post("/upload/")
@limiter.limit("5/day")
async def upload(request: Request, files: List[UploadFile] = File(...)):
    max_files_per_request = 5
    ip = request.client.host
    count = increment_counter(upload_counter, ip, 5)

    if len(files) > max_files_per_request:
        return {
            "error": f"❌ Upload failed. You can upload up to {max_files_per_request} files at a time.",
            "num_files": 0
        }

    result = await load_docs(files)
    return {"num_files": len(files), "usage": f"{count}/5"}

@app.post("/query/")
@limiter.limit("15/day")
async def query(request: Request, question: str = Form(...)):
    ip = request.client.host
    count = increment_counter(query_counter, ip, 20)

    docs = get_loaded_docs()
    response = bot(docs, question)
    if not response:
        return {"error": "No reply from bot"}
    
    return {"reply": response, "usage": f"{count}/20"}

import shutil

@app.on_event("shutdown")
def cleanup_temp_files():
    tmp_dir = "tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)