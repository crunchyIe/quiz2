import PyPDF2
import re
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        # Clean up the text
        text = clean_text(text)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return "Error extracting text from PDF."

def extract_text_from_txt(txt_path):
    """Extract text from a TXT file"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Clean up the text
        text = clean_text(text)
        return text
    except UnicodeDecodeError:
        # Try different encoding if utf-8 fails
        try:
            with open(txt_path, 'r', encoding='latin-1') as file:
                text = file.read()
            
            # Clean up the text
            text = clean_text(text)
            return text
        except Exception as e:
            logger.error(f"Error reading TXT file with latin-1 encoding {txt_path}: {str(e)}")
            return "Error extracting text from TXT file."
    except Exception as e:
        logger.error(f"Error extracting text from TXT {txt_path}: {str(e)}")
        return "Error extracting text from TXT file."

def clean_text(text):
    """Clean and normalize extracted text"""
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    # Remove page numbers (common in PDFs)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Remove header/footer artifacts (optional - may need customization)
    # text = re.sub(r'Header|Footer', '', text)
    
    return text.strip()
