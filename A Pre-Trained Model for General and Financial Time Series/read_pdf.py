#!/usr/bin/env python3
"""
Script to read and analyze the PDF paper content
"""

import PyPDF2
import sys
import re
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            print(f"PDF has {len(pdf_reader.pages)} pages")
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text
                
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def analyze_paper_content(text):
    """Analyze the paper content and extract key information"""
    if not text:
        return
    
    print("\n" + "="*80)
    print("PAPER ANALYSIS")
    print("="*80)
    
    # Extract title (usually in the first few lines)
    lines = text.split('\n')
    first_lines = [line.strip() for line in lines[:20] if line.strip()]
    
    print("\n📄 TITLE AND BEGINNING:")
    for i, line in enumerate(first_lines[:10]):
        if line and len(line) > 10:
            print(f"  {line}")
    
    # Look for abstract
    abstract_match = re.search(r'Abstract\s*[:\-]?\s*(.*?)(?=\n\s*\n|\n\s*1\s+Introduction|\n\s*Keywords)', text, re.DOTALL | re.IGNORECASE)
    if abstract_match:
        abstract = abstract_match.group(1).strip()
        print(f"\n📋 ABSTRACT:")
        print(f"  {abstract[:1000]}{'...' if len(abstract) > 1000 else ''}")
    
    # Look for keywords
    keywords_match = re.search(r'Keywords?\s*[:\-]?\s*(.*?)(?=\n\s*\n|\n\s*1)', text, re.DOTALL | re.IGNORECASE)
    if keywords_match:
        keywords = keywords_match.group(1).strip()
        print(f"\n🔑 KEYWORDS:")
        print(f"  {keywords}")
    
    # Extract section headers (look for numbered sections)
    section_pattern = r'\n\s*(\d+(?:\.\d+)*)\s+([A-Z][^\n]+)'
    sections = re.findall(section_pattern, text)
    
    if sections:
        print(f"\n📚 MAIN SECTIONS:")
        for num, title in sections[:15]:  # Show first 15 sections
            print(f"  {num} {title.strip()}")
    
    # Look for methodology/approach keywords
    method_keywords = ['model', 'algorithm', 'method', 'approach', 'framework', 'architecture', 'neural', 'deep learning', 'machine learning', 'transformer', 'attention']
    found_methods = []
    text_lower = text.lower()
    for keyword in method_keywords:
        if keyword in text_lower:
            found_methods.append(keyword)
    
    if found_methods:
        print(f"\n🔬 METHODOLOGY KEYWORDS FOUND:")
        print(f"  {', '.join(found_methods)}")
    
    # Look for datasets mentioned
    dataset_pattern = r'(?:dataset|data set|corpus|benchmark)\s*[:\-]?\s*([A-Z][A-Za-z0-9\-_\s]+?)(?=\s*[,.\n])'
    datasets = re.findall(dataset_pattern, text, re.IGNORECASE)
    if datasets:
        unique_datasets = list(set([d.strip() for d in datasets if len(d.strip()) > 3]))[:10]
        print(f"\n📊 DATASETS/BENCHMARKS MENTIONED:")
        for dataset in unique_datasets:
            print(f"  - {dataset}")
    
    # Look for results/performance
    result_keywords = ['accuracy', 'performance', 'results', 'evaluation', 'experiment', 'baseline', 'state-of-the-art', 'SOTA']
    print(f"\n📈 PERFORMANCE/RESULTS INDICATORS:")
    for keyword in result_keywords:
        count = text_lower.count(keyword.lower())
        if count > 0:
            print(f"  - '{keyword}' mentioned {count} times")

def main():
    pdf_path = "2506.06288v1.pdf"
    
    if not Path(pdf_path).exists():
        print(f"Error: PDF file '{pdf_path}' not found!")
        return
    
    print(f"Reading PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    
    if text:
        analyze_paper_content(text)
        
        # Save extracted text for further analysis if needed
        with open("extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n💾 Full extracted text saved to 'extracted_text.txt'")
    else:
        print("Failed to extract text from PDF")

if __name__ == "__main__":
    main()
