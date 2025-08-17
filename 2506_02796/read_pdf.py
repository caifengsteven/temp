#!/usr/bin/env python3
"""
Script to read and extract text from PDF file
"""

import sys
import os

def read_pdf_with_pypdf2():
    """Try reading PDF with PyPDF2"""
    try:
        import PyPDF2
        
        with open('2506.02796v1.pdf', 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            print(f"Number of pages: {len(pdf_reader.pages)}")
            print("=" * 50)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text
                
            return text
    except ImportError:
        print("PyPDF2 not available")
        return None
    except Exception as e:
        print(f"Error with PyPDF2: {e}")
        return None

def read_pdf_with_pdfplumber():
    """Try reading PDF with pdfplumber"""
    try:
        import pdfplumber
        
        text = ""
        with pdfplumber.open('2506.02796v1.pdf') as pdf:
            print(f"Number of pages: {len(pdf.pages)}")
            print("=" * 50)
            
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text
                    
        return text
    except ImportError:
        print("pdfplumber not available")
        return None
    except Exception as e:
        print(f"Error with pdfplumber: {e}")
        return None

def read_pdf_with_pymupdf():
    """Try reading PDF with PyMuPDF (fitz)"""
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open('2506.02796v1.pdf')
        text = ""
        
        print(f"Number of pages: {len(doc)}")
        print("=" * 50)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page_text
            
        doc.close()
        return text
    except ImportError:
        print("PyMuPDF not available")
        return None
    except Exception as e:
        print(f"Error with PyMuPDF: {e}")
        return None

def main():
    if not os.path.exists('2506.02796v1.pdf'):
        print("PDF file '2506.02796v1.pdf' not found!")
        return
    
    print("Attempting to read PDF file: 2506.02796v1.pdf")
    print("=" * 60)
    
    # Try different PDF libraries
    text = None
    
    # Try PyMuPDF first (usually most reliable)
    print("Trying PyMuPDF...")
    text = read_pdf_with_pymupdf()
    
    if not text:
        print("\nTrying pdfplumber...")
        text = read_pdf_with_pdfplumber()
    
    if not text:
        print("\nTrying PyPDF2...")
        text = read_pdf_with_pypdf2()
    
    if text:
        print("\n" + "=" * 60)
        print("PDF CONTENT:")
        print("=" * 60)
        print(text)
        
        # Save to text file for easier reading
        with open('pdf_content.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"\nContent saved to 'pdf_content.txt'")
    else:
        print("\nFailed to read PDF with all available libraries.")
        print("You may need to install one of these packages:")
        print("  pip install PyMuPDF")
        print("  pip install pdfplumber") 
        print("  pip install PyPDF2")

if __name__ == "__main__":
    main()
