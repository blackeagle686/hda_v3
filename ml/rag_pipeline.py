import os
import redis
import chromadb
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class MedicalRAG:
    def __init__(self, 
                 chroma_path: str = "chroma_db", 
                 collection_name: str = "medical_docs",
                 redis_url: str = "redis://localhost:6379/0",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the RAG system.
        """
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        
        # Initialize Embeddings
        # Use GPU if available for maximum performance on server
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading Embedding Model ({device}): {embedding_model}")
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device}
        )
        
        # Initialize Vector Store (ChromaDB)
        self.vector_store = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embedding_function,
            collection_name=self.collection_name
        )
        
        # Initialize Redis
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            print("[INFO] Connected to Redis.")
        except redis.ConnectionError:
            print("[WARN] Redis connection failed. Caching will be disabled.")
            self.redis_client = None

    def ingest_documents(self, source_folder: str = "DATA"):
        """
        Load PDFs, chunk them, and store in ChromaDB.
        """
        if not os.path.exists(source_folder):
            print(f"[ERROR] Folder {source_folder} not found.")
            return

        print(f"[INFO] Scanning {source_folder} for PDFs...")
        pdf_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print("[INFO] No PDFs found.")
            return

        documents = []
        for pdf in pdf_files:
            file_path = os.path.join(source_folder, pdf)
            print(f"[INFO] Loading {pdf}...")
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"[ERROR] Failed to load {pdf}: {e}")

        if not documents:
            return

        # Split Text
        print("[INFO] Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[INFO] Created {len(chunks)} chunks.")

        # Add to Chroma
        print("[INFO] Adding to Vector Store...")

        batch_size = 5000   # أقل من الحد الأقصى
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.vector_store.add_documents(batch)
            print(f"[INFO] Added batch {i//batch_size + 1}")

        print("[INFO] Ingestion Complete.")

    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        """
        if not self.vector_store:
            print("[WARN] Vector store not initialized.")
            return []
            
        print(f"[INFO] Searching for: {query}")
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"[ERROR] RAG search failed: {e}")
            return []

    def format_context(self, docs: List[Document]) -> str:
        """
        Format retrieved documents into a single string for the LLM.
        """
        context = "\n\n".join([f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}" for d in docs])
        return context

# Utilities for manual testing
if __name__ == "__main__":
    rag = MedicalRAG()
    # ingest manually
    # rag.ingest_documents()
    # search manually
    # res = rag.search("Lung adenocarcinoma treatment")
    # print(rag.format_context(res))
