import PyPDF2
import os
import glob

def read_pdf(file_path):
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def main():
    # Find all PDF files in current directory
    pdf_files = glob.glob("*.pdf")
    
    print(f"Found {len(pdf_files)} PDF files:")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"{i}. {pdf_file}")
    
    print("\n" + "="*80 + "\n")
    
    # Read each PDF and extract content
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"STRATEGY {i}: {pdf_file}")
        print("="*80)
        
        content = read_pdf(pdf_file)
        if content:
            # Save content to text file for easier analysis
            txt_filename = f"strategy_{i}_content.txt"
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Content saved to {txt_filename}")
            
            # Print first 2000 characters for preview
            print("\nContent preview:")
            print("-" * 40)
            print(content[:2000])
            if len(content) > 2000:
                print("\n... (content truncated, see full content in text file)")
        else:
            print("Failed to extract content from this PDF")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
