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
def _log_msg(msg, func_name): 
    print(f'---------{func_name}-----------------')
    print(msg)
    print(f'----------{func_name}----------------')
    
# Initialize App
app = FastAPI(title="Health Data Analysis AI Assistant")

# Configuration
USE_GEMINI = os.getenv("USE_GEMINI", "True").lower() == "true"
MOCK_LLM = False

# Initialize RAG System
rag_system = MedicalRAG() 

# Initialize Pipeline
# In production/cloud, use Gemini to avoid local hardware constraints
pipeline = HDAPipeline(
    model_path="checkpoints/best_model.pth", 
    mock_llm=MOCK_LLM,
    use_gemini=USE_GEMINI,
    rag_system=rag_system
)

from ml.report_gen import ReportGenerator
from fastapi import BackgroundTasks

# Directories
UPLOAD_DIR = "static/uploads"
REPORTS_DIR = "static/reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

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
        _log_msg(f"user message: {message}\ncontext_summarise: {context_summaries}\nsessionID:{session_id}", func_name="endpoint of: /api/chat")
        # Parse context summaries
        summaries = json.loads(context_summaries)
        
        # RAG Retrieval
        rag_context = ""
        sources = []
        try:
            docs = rag_system.search(message)
            rag_context = rag_system.format_context(docs)
            sources = rag_system.get_sources(docs)
            
            if rag_context:
                print(f"[INFO] RAG Context found: {len(docs)} docs")
        except Exception as e:
            print(f"[WARN] RAG search failed: {e}")
            
        # Chat with Unified Qwen
        result = pipeline.chat(message, summaries, rag_context)
        
        # Append sources to the response text
        if sources:
            result["response"] += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])
        
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_file(path: str):
    """Utility to delete a file after it has been sent."""
    if os.path.exists(path):
        os.remove(path)

@app.post("/api/download-report")
async def download_report_endpoint(
    content: str = Form(...),
    format: str = Form("pdf"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Generates and returns a downloadable report file.
    """
    unique_id = str(uuid.uuid4())
    filename = f"hda_report_{unique_id}.{format}"
    file_path = os.path.join(REPORTS_DIR, filename)

    try:
        if format == "pdf":
            ReportGenerator.to_pdf(content, file_path)
        elif format == "docx":
            ReportGenerator.to_word(content, file_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid format")

        # Cleanup file after response is sent
        background_tasks.add_task(cleanup_file, file_path)
        
        return FileResponse(
            file_path, 
            filename=f"HDA_Medical_Report.{format}",
            media_type="application/octet-stream"
        )
    except Exception as e:
        print("DOWNLOAD ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    message: str = Form("")
):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Invalid file type"})

    file_ext = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # analyze_image returns { "classification": ..., "advice": ..., "summary": ..., "sources": ... }
        result = pipeline.analyze_image(file_path, message)

        mapped_class = LABEL_MAP.get(result["classification"]["class"], "Unknown")
        
        advice_text = result["advice"]
        if result.get("sources"):
            advice_text += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in result["sources"]])

        return JSONResponse(
            content={
                "classification": {
                    "class": mapped_class,
                    "confidence": float(result["classification"]["confidence"])
                },
                "response": advice_text,
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
