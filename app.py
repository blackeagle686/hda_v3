from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ml.pipeline import HDAPipeline
from ml.rag_pipeline import MedicalRAG
import shutil
from dotenv import load_dotenv
import os
import uuid

# Load environment variables
load_dotenv()

# Initialize App
app = FastAPI(title="Health Data Analysis AI Assistant")

# Configuration
USE_GEMINI = os.getenv("USE_GEMINI", "True").lower() == "true"
MOCK_LLM = os.getenv("MOCK_LLM", "False").lower() == "true"

# Initialize Pipeline
# In production/cloud, use Gemini to avoid local hardware constraints
pipeline = HDAPipeline(
    model_path="checkpoints/best_model.pth", 
    mock_llm=MOCK_LLM,
    use_gemini=USE_GEMINI
)
rag_system = MedicalRAG() # Initialize RAG System

# Directories
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
LABEL_MAP = {
    'colon_aca': 'Colon Adenocarcinoma',
    'colon_n': 'Colon Normal',
    'lung_aca': 'Lung Adenocarcinoma',
    'lung_n': 'Lung Normal',
    'lung_scc': 'Lung Squamous Cell Carcinoma'
}

# Mount Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("templates/index.html")

# Global in-memory chat history storage
CHAT_HISTORY = {}

import json
from typing import List, Optional

@app.post("/api/chat")
async def chat_endpoint(
    message: str = Form(...), 
    context_summaries: str = Form("[]"), # Received as JSON string from frontend
    session_id: str = Form(...)
):
    try:
        # Parse context summaries
        summaries = json.loads(context_summaries)
        
        # RAG Retrieval
        rag_context = ""
        try:
            docs = rag_system.search(message)
            rag_context = rag_system.format_context(docs)
            if rag_context:
                print(f"[INFO] RAG Context found: {len(docs)} docs")
        except Exception as e:
            print(f"[WARN] RAG search failed: {e}")
            
        # Chat with Unified Qwen
        # Returns { "response": str, "summary": str }
        result = pipeline.chat(message, summaries, rag_context)
        
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Invalid file type"})

    file_ext = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # analyze_image returns { "classification": ..., "advice": ..., "summary": ... }
        result = pipeline.analyze_image(file_path)

        mapped_class = LABEL_MAP.get(result["classification"]["class"], "Unknown")
        
        return JSONResponse(
            content={
                "classification": {
                    "class": mapped_class,
                    "confidence": float(result["classification"]["confidence"])
                },
                "response": result["advice"], # Frontend expects 'advice' mapped to response usually, but we'll use 'response' key for consistency if we change JS, or keep 'advice'
                "summary": result["summary"],
                "original_url": f"/static/uploads/{unique_filename}",
            }
        )

    except Exception as e:
        print("ANALYSIS ERROR:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
