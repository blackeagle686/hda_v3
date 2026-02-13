# API Endpoints Documentation

All API endpoints are defined in [`app.py`](file:///c:/Users/The_Last_King/OneDrive/Documents/Projects/myAiTools/HDA/hda_v3/app.py).

## Base URL
The default development server runs at `http://localhost:8000`

## Endpoints List

### 1. Root / UI
- **URL:** `/`
- **Method:** `GET`
- **Description:** Serves the main web interface from `templates/index.html`.
- **Response:** HTML File.

### 2. Chat API
- **URL:** `/api/chat`
- **Method:** `POST`
- **Description:** Session-based chat with the AI medical assistant. Integrates RAG to provide evidence-based answers.
- **Parameters (Form Data):**
    - `message` (str): The user's question or input.
    - `context` (str): Optional background information for the LLM.
    - `session_id` (str): Unique identifier for the chat session (for history persistence).
- **Response:** JSON
    ```json
    { "response": "Assistant reply text..." }
    ```

### 3. Analyze API
- **URL:** `/api/analyze`
- **Method:** `POST`
- **Description:** Uploads a medical image for pathology classification and clinical report generation. Enriched with RAG-retrieved medical context.
- **Parameters (Multipart/Form):**
    - `file`: The image file (JPG, PNG, etc.).
- **Response:** JSON
    ```json
    {
        "classification": {
            "class": "Lung Adenocarcinoma",
            "confidence": 0.98
        },
        "advice": "VLM generated clinical advice... [Additional Reference Information]...",
        "original_url": "/static/uploads/unique_id.png"
    }
    ```

### 4. Static Files
- **URL:** `/static/*`
- **Method:** `GET`
- **Description:** Mounts and serves files from the `static/` directory (uploads, styles, scripts).
