from ml.rag_pipeline import MedicalRAG
import sys

def main():
    rag = MedicalRAG()
    print("Starting ingestion...")
    rag.ingest_documents("DATA")
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
