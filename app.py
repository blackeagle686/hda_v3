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
pipeline = HDAPipeline(model_path="checkpoints/best_model.pth", mock_llm=False)

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

        return JSONResponse(
            content={
                "classification": {
                    "class": result["classification"]["class"],
                    "confidence": float(result["classification"]["confidence"])
                },
                "advice": result["advice"],
                "original_url": f"/static/uploads/{unique_filename}"
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
