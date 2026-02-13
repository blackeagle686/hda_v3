from fpdf import FPDF
import os

def create_dummy_pdf(filename="DATA/medical_sample.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    text = """
    Medical Reference: Lung Cancer Analysis
    
    Lung adenocarcinoma is the most common subtype of non-small cell lung cancer (NSCLC).
    It typically arises in the outer periphery of the lungs.
    
    Key Features:
    - Glandular formation
    - Mucin production
    - TTF-1 positive in immunohistochemistry
    
    Treatment:
    - Surgical resection for early stages.
    - Chemotherapy and immunotherapy for advanced stages.
    - Targeted therapy for EGFR notations.
    
    Colon Adenocarcinoma:
    This is a cancer of the mucus-secreting glands of the colon.
    Screening is typically done via colonoscopy.
    """
    
    pdf.multi_cell(0, 10, text)
    
    os.makedirs("DATA", exist_ok=True)
    pdf.output(filename)
    print(f"Created {filename}")

if __name__ == "__main__":
    create_dummy_pdf()
