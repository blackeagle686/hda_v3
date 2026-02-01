from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ml.pipeline import HDAPipeline
import shutil
import os
import uuid

# Initialize App
app = FastAPI(title="Health Data Analysis AI Assistant")

# Initialize Pipeline (Mock by default for local dev)
# In production, set mock_llm=False and provide model_path
pipeline = HDAPipeline(model_path="checkpoints/best_model.pth", mock_llm=True)

# Directories
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("templates/index.html")

@app.post("/api/chat")
async def chat_endpoint(message: str = Form(...), context: str = Form("")):
    try:
        response = pipeline.chat(message, context)
        return JSONResponse({"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    # Save uploaded file
    file_ext = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Generate Heatmap Path
    heatmap_filename = f"heatmap_{unique_filename}"
    heatmap_path = os.path.join(UPLOAD_DIR, heatmap_filename)
    
    try:
        # Run Analysis
        result = pipeline.analyze_image(file_path, heatmap_path)
        
        # Construct URLs for frontend
        result["heatmap_url"] = f"/static/uploads/{heatmap_filename}"
        result["original_url"] = f"/static/uploads/{unique_filename}"
        
        return JSONResponse(result)
        
    except Exception as e:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
