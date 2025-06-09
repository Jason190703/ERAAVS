import io
import PyPDF2
import pdfplumber
import numpy as np
import tempfile
import os
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file using both PyPDF2 and pdfplumber.
    Falls back to the alternative method if one fails.
    
    Args:
        pdf_file: Uploaded PDF file object
    
    Returns:
        str: Extracted text from the PDF
    """
    text = ""
    
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.read())
        temp_path = temp_file.name
    
    try:
        # First try using PyPDF2
        try:
            with open(temp_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Check if the PDF is encrypted
                if reader.is_encrypted:
                    try:
                        reader.decrypt('')  # Try empty password
                    except:
                        logger.warning("Could not decrypt the PDF with PyPDF2")
                        raise Exception("Encrypted PDF")
                
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                if not text.strip():
                    raise Exception("No text extracted with PyPDF2")
                    
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {str(e)}")
            # If PyPDF2 fails, we'll try pdfplumber next
            raise
        
    except Exception as e:
        # Try using pdfplumber if PyPDF2 fails
        logger.info("Falling back to pdfplumber for text extraction")
        text = ""
        
        try:
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e2:
            logger.error(f"pdfplumber extraction also failed: {str(e2)}")
            text = f"Error extracting text from PDF: {str(e2)}"
    
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file: {str(e)}")
    
    # Post-process the text
    text = clean_text(text)
    
    return text

def clean_text(text):
    """
    Clean and normalize extracted text
    
    Args:
        text (str): Text to clean
    
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple newlines with a single one
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with a single one
    text = re.sub(r' +', ' ', text)
    
    # Remove non-breaking spaces and other special whitespace
    text = text.replace('\xa0', ' ')
    
    # Remove unnecessary punctuation that might have been added during extraction
    text = re.sub(r'(?<!\d)\.(?!\d|\w)', '. ', text)  # Add space after periods not in numbers
    text = re.sub(r'\s+\.', '.', text)  # Remove spaces before periods
    
    # Standardize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text.strip()
