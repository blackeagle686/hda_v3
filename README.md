# 🏥 Health Data Analysis (HDA) AI Assistant - V3

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Revolutionizing Histopathology with AI-Powered Intelligence.**
> HDA V3 is an advanced medical diagnostic assistant that combines Deep Learning classification with Large Vision-Language Models (VLM) and RAG to provide comprehensive analysis of medical slides.

---

## 💡 The Idea & Purpose

The **Health Data Analysis (HDA) AI Assistant** is designed to empower medical professionals by providing a "second-opinion" screening tool for histopathology. By automating the identification of specific cancer types in lung and colon tissues, HDA reduces the diagnostic workflow latency and provides descriptive, evidence-based clinical reports.

### Key Features
-   🔬 **Precision Classification**: Detects Colon Adenocarcinoma, Lung Adenocarcinoma, and Squamous Cell Carcinoma with high confidence using a fine-tuned EfficientNet-B0.
-   📝 **AI-Generated Clinical Reports**: Leverages **Qwen2-VL** to generate natural language descriptions of visual pathological features.
-   💬 **Interactive Medical Chat**: Session-based AI consultation powered by **TinyLlama/Gemini** for follow-up questions.
-   📚 **Medical RAG System**: Integrates clinical guidelines and research papers using **ChromaDB** to provide cited, evidence-based answers.
-   🔥 **Visual Evidence**: Generates Grad-CAM heatmaps to highlight critical regions within pathology slides.

---

## 🛠 Tech Stack

### 🤖 AI & Machine Learning

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-5A5AFA?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/Accelerate-FF6F00?style=for-the-badge)

Vision:
EfficientNet-B0 • Qwen2-VL

LLM:
TinyLlama • Google Gemini

---

### 🏗 Backend & Deployment

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-499848?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

### 🎨 Frontend

![HTML](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
---

## 🏗 Architecture Flow

The system operates as a modular pipeline where data flows from user input to comprehensive diagnostic output.

```mermaid
graph TD
    A[User Uploads Image] --> B{FastAPI Backend}
    B --> C[HDAPipeline Orchestrator]
    
    subgraph "AI Analysis"
    C --> D[EfficientNet Classifier]
    C --> E[Qwen2-VL Reporter]
    C --> F[Medical RAG System]
    end
    
    D --> G[Classification & Confidence]
    E --> H[Clinical Description]
    F --> I[Contextual References]
    
    G & H & I --> J[Aggregated Analysis Result]
    J --> K[Interactive Chat Interface]
    J --> L[PDF/Word Report Download]
```

---

## 🚀 Getting Started

### Prerequisites
-   Python 3.9+
-   CUDA-compatible GPU (Recommended for local inference) or Gemini API Key.
-   16GB+ RAM.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/HDA_V3.git
    cd HDA_V3
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Environment**:
    Create a `.env` file based on `.env.example`:
    ```env
    USE_GEMINI=True
    GOOGLE_API_KEY=your_key_here
    ```

### Running the App

```bash
python app.py
```
*Access the dashboard at `http://localhost:8000`*

---

## 📅 Roadmap

-   [ ] **V4 Expansion**: Support for X-Ray, MRI, and CT scan classification.
-   [ ] **Multi-Modal Integration**: Combined analysis of patient history (EHR) and images.
-   [ ] **Deployment**: Dockerization and Kubernetes scaling for clinical environments.

---

## 📄 Documentation

Detailed documentation can be found in the [`DOC/`](DOC/) directory:
-   [Software Requirements (SRS)](DOC/SRS.md)
-   [Architecture & Class Definitions](DOC/classes_and_pipelines.md)
-   [API Endpoints](DOC/endpoints.md)

---

Developed with ❤️ for the Medical Community.
