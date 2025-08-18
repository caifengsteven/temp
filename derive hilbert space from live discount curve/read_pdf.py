import PyPDF2
import sys

def read_pdf(filename):
    try:
        with open(filename, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            print(f'Number of pages: {len(pdf_reader.pages)}')
            print('\n' + '='*50)
            print('EXTRACTING TEXT FROM PDF:')
            print('='*50 + '\n')
            
            full_text = ''
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                full_text += text + '\n\n'
                print(f'--- Page {page_num + 1} ---')
                print(text)
                print('\n')
                
            return full_text
            
    except Exception as e:
        print(f'Error reading PDF: {e}')
        return None

if __name__ == "__main__":
    text = read_pdf('2506.03342v1.pdf')
