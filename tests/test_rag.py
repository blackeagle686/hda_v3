import unittest
import os
import shutil
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.rag_pipeline import MedicalRAG

class TestMedicalRAG(unittest.TestCase):
    def setUp(self):
        # Use a temporary ChromaDB path for testing
        self.test_chroma_path = "test_chroma_db"
        self.rag = MedicalRAG(chroma_path=self.test_chroma_path)
        
        # Ensure DATA folder exists
        os.makedirs("DATA", exist_ok=True)
        # Create a tiny dummy PDF if not exists (using manual create just in case)
        # We'll just rely on create_test_pdf.py having run, or create a text file mimicking PDF logic if we mocked it,
        # but since we use PyPDFLoader, we need a real PDF.
        # So we assume create_test_pdf.py works.

    def tearDown(self):
        # Cleanup
        if os.path.exists(self.test_chroma_path):
            shutil.rmtree(self.test_chroma_path)

    def test_ingestion_and_search(self):
        # 1. Ingest
        print("\nTesting Ingestion...")
        self.rag.ingest_documents("DATA")
        
        # 2. Search
        print("Testing Search...")
        results = self.rag.search("Lung cancer treatment", k=1)
        
        # 3. Verify
        self.assertTrue(len(results) > 0, "Should return at least one result")
        print(f"Retrieved: {results[0].page_content[:100]}...")
        self.assertIn("Lung", results[0].page_content, "Content should match query context")

if __name__ == "__main__":
    unittest.main()
