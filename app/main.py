from fastapi import FastAPI, UploadFile, File
from typing import List, Annotated
import shutil
from app.rag import retrieve_relevant_chunks
from app.rag import (
    extract_text_from_pdf,
    chunk_text,
    store_chunks
)
from app.rag import generate_answer
app = FastAPI()

@app.get("/")
def home():
    return {"message": "RAG Platform Running"}

@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):

    processed_files = []

    for file in files:

        file_path = f"uploads/{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_text = extract_text_from_pdf(file_path)

        chunks = chunk_text(extracted_text)

        store_chunks(chunks, file.filename)

        processed_files.append({
            "filename": file.filename,
            "chunks": len(chunks)
        })

    return {
        "message": "Documents processed successfully",
        "files": processed_files
    }

@app.get("/search")
def search(query: str):

    results = retrieve_relevant_chunks(query)

    return {
        "query": query,
        "results": results
    }

@app.get("/chat")
def chat(query: str):

    result = generate_answer(query)

    return {
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"]
    }