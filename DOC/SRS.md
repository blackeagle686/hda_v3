# Software Requirements Specification (SRS) - Health Data Analysis (HDA)

## 1. Introduction
The **Health Data Analysis AI Assistant (HDA)** is an advanced system designed to assist medical professionals in analyzing histopathology images. By leveraging state-of-the-art Deep Learning (EfficientNet) and Large Language/Vision Models (TinyLlama & Qwen2-VL), the system provides classification, automated clinical reporting, and an interactive chat interface for medical consultation.

## 2. Idea & Purpose
The primary goal of HDAs is to reduce the workload of pathologists and doctors by providing a "second opinion" or initial screening of medical slides. It specializes in detecting lung and colon cancers and providing human-like explanations and advice based on visual evidence.

## 3. Functional Requirements (FR)
- **FR1: Image Upload & Preprocessing:** The system shall allow users to upload medical images (JPG, PNG) and automatically resize/normalize them for analysis.
- **FR2: Pathology Classification:** The system shall classify images into five distinct categories:
    - Colon Adenocarcinoma
    - Colon Normal
    - Lung Adenocarcinoma
    - Lung Normal
    - Lung Squamous Cell Carcinoma
- **FR3: Confidence Scoring:** The system shall provide a probability/confidence score for each classification.
- **FR4: Vision-Language Reporting:** The system shall generate a detailed clinical report explaining visual features in the image using a VLM.
- **FR5: Medical AI Chat:** The system shall provide a session-based chat interface for users to ask follow-up questions regarding the analysis or general symptoms.
- **FR6: Grad-CAM Visualization:** (Available in backend) The system should be capable of generating heatmaps to highlight areas of interest in the image.
- **FR7: Model Training:** The system shall include a training pipeline for fine-tuning the classification model on new datasets.

## 4. Non-Functional Requirements (NFR)
- **NFR1: Performance:** Image analysis and report generation should ideally complete within 2-5 seconds (depending on hardware/GPU availability).
- **NFR2: Scalability:** The system uses FastAPI and modular ML components to allow for independent scaling of services.
- **NFR3: Accuracy:** The system targets high sensitivity and specificity for cancer detection (verified during training phases).
- **NFR4: Security:** In-memory chat history ensures session privacy, though clinical data should always be handled with strict compliance.
- **NFR5: Compatibility:** The backend supports both CPU and CUDA-enabled GPU environments.

## 5. Future Roadmap
The following features are planned for future development to enhance the HDA system's reliability and clinical utility:

- **RAG (Retrieval-Augmented Generation) System:** Integration of a vector database containing clinical guidelines, latest medical research, and pathology textbooks. This will allow the AI assistant to provide evidence-based responses with citations.
- **Support for More Medical Image Types:** Expanding the classification and analysis capabilities to include X-rays, MRI scans, and CT scans, providing a more comprehensive diagnostic tool.
- **Backend Integration for Doctors & Clinics:** Connecting the HDA system with existing clinic management systems (ERPs) and electronic health records (EHR). This will allow the system to cross-reference patient history with image analysis for higher reliability and personalized advice.
