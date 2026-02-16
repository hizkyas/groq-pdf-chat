import os
import shutil
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.services import process_pdf, query_vector_db

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

class QueryRequest(BaseModel):
    prompt: str

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        num_chunks = process_pdf(temp_path)
        return {"status": "success", "chunks": int(num_chunks)}
    except Exception as e:
        # This will print the exact error line in your terminal
        print(traceback.format_exc())
        return {"status": "error", "message": str(e), "chunks": 0}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        answer = query_vector_db(request.prompt)
        # We ensure it's ALWAYS a string
        return {"response": str(answer)}
    except Exception as e:
        print(traceback.format_exc())
        # If Python fails, we send the error back as the response!
        return {"response": f"Python Error: {str(e)}"}

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")