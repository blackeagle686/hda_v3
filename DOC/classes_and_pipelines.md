# System Classes and Pipelines

This document provides a detailed overview of the core classes and pipelines that drive the HDA system.

## ML Architecture & Pipelines

### 1. `HDAPipeline`
- **Location:** [`ml/pipeline.py`](file:///c:/Users/The_Last_King/OneDrive/Documents/Projects/myAiTools/HDA/hda_v3/ml/pipeline.py)
- **Role:** The main orchestrator of the system. It integrates the image classifier, LLM, and VLM into a single workflow.
- **Key Methods:**
    - `analyze_image(image_path)`: Runs classification and generates a VLM report.
    - `chat(message, history, context)`: Interacts with the LLM for chat functionality.

### 2. `HDAImgClassifier`
- **Location:** [`ml/image_model.py`](file:///c:/Users/The_Last_King/OneDrive/Documents/Projects/myAiTools/HDA/hda_v3/ml/image_model.py)
- **Role:** Handles image classification using a fine-tuned EfficientNet-B0 model.
- **Key Methods:**
    - `predict(image_path)`: Preprocesses image and returns class and confidence.
    - `generate_heatmap(image_path, save_path)`: Uses Grad-CAM to generate activation heatmaps.

### 3. `LLMResponder`
- **Location:** [`ml/llm_pipeline.py`](file:///c:/Users/The_Last_King/OneDrive/Documents/Projects/myAiTools/HDA/hda_v3/ml/llm_pipeline.py)
- **Role:** Manages text-based AI responses using `TinyLlama`. Supports a mock mode for local development.
- **Key Methods:**
    - `generate_response(user_text, context, history)`: Formats prompts and retrieves model responses.

### 4. `VLMReporter`
- **Location:** [`ml/llm_pipeline.py`](file:///c:/Users/The_Last_King/OneDrive/Documents/Projects/myAiTools/HDA/hda_v3/ml/llm_pipeline.py)
- **Role:** Uses `Qwen2-VL` to analyze images and classification results to produce descriptive clinical reports.
- **Key Methods:**
    - `generate_report(image_path, classification_result)`: Generates natural language analysis of images.

### 5. `HDAImgTrainer`
- **Location:** [`my_model.py`](file:///c:/Users/The_Last_King/OneDrive/Documents/Projects/myAiTools/HDA/hda_v3/my_model.py)
- **Role:** A dedicated training class for fine-tuning EfficientNet-B0 on histopathology datasets.
- **Key Methods:**
    - `train(epochs)`: Executes the training loop.
    - `evaluate()`: Validates model performance on a test/validation set.

### 6. `MedicalRAG`
- **Location:** [`ml/rag_pipeline.py`](file:///c:/Users/The_Last_King/OneDrive/Documents/Projects/myAiTools/HDA/hda_v3/ml/rag_pipeline.py)
- **Role:** Manage RAG operations including document ingestion, vector storage (ChromaDB), and semantic search.
- **Key Methods:**
    - `ingest_documents(source_folder)`: Loads PDFs, chunks them, and stores embeddings.
    - `search(query)`: Retrieves relevant document chunks.
    - `format_context(docs)`: Formats retrieved docs for LLM context.

## Web Application Logic

### FastAPI App Instance
- **Location:** [`app.py`](file:///c:/Users/The_Last_King/OneDrive/Documents/Projects/myAiTools/HDA/hda_v3/app.py)
- **Role:** Main entry point for the web server, handles routing, static file serving, and pipeline initialization.
