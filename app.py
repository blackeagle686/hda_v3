from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ml.pipeline import HDAPipeline
from ml.rag_pipeline import MedicalRAG
import shutil
import os
import uuid

# Initialize App
app = FastAPI(title="Health Data Analysis AI Assistant")

# Initialize Pipeline (Mock by default for local dev)
# In production, set mock_llm=False and provide model_path
pipeline = HDAPipeline(model_path="checkpoints/best_model.pth", mock_llm=False)
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

@app.post("/api/chat")
async def chat_endpoint(message: str = Form(...), context: str = Form(""), session_id: str = Form(...)):
    try:
        # Get history for this session
        history = CHAT_HISTORY.get(session_id, [])
        
        # RAG Retrieval
        rag_context = ""
        try:
            # Use the global RAG instance
            docs = rag_system.search(message)
            rag_context = rag_system.format_context(docs)
            if rag_context:
                print(f"[INFO] RAG Context found for chat: {len(docs)} docs")
        except Exception as e:
            print(f"[WARN] RAG search failed: {e}")
            
        # Combine user context with RAG context
        full_context = f"{context}\n\nRelevant Medical Info:\n{rag_context}" if rag_context else context

        response = pipeline.chat(message, history, full_context)
        
        # Update history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        CHAT_HISTORY[session_id] = history
        
        return JSONResponse({"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid file type"}
        )

    file_ext = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = pipeline.analyze_image(file_path)

        mapped_class = LABEL_MAP.get(result["classification"]["class"], "Unknown")
        
        # RAG Enhancement for Advice using Classification result
        rag_advice = ""
        try:
            # Search for the classified disease
            query = f"Treatment and features of {mapped_class}"
            docs = rag_system.search(query)
            rag_context = rag_system.format_context(docs)
            
            if rag_context and result.get("advice"):
                # Append RAG info to the VLM advice or refine it
                # For now, we'll append it to the advice string
                rag_advice = f"\n\n[Additional Reference Information]:\n{rag_context[:500]}..." # Truncate for brevity in JSON
        except Exception as e:
            print(f"[WARN] RAG advice generation failed: {e}")

        final_advice = result["advice"] + rag_advice

        return JSONResponse(
            content={
                "classification": {
                    "class": mapped_class,
                    "confidence": float(result["classification"]["confidence"])
                },
                "advice": final_advice,
                "original_url": f"/static/uploads/{unique_filename}",
            }
        )

    except Exception as e:
        print("ANALYSIS ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
