import PyPDF2
from io import BytesIO

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text content from PDF file."""
    pdf_file = BytesIO(file_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    return text.strip()

