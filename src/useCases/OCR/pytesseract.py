import pytesseract
import PyPDF2

# Set the path to the PDF file
pdf_file = 'example.pdf'

# Open the PDF file using PyPDF2
with open(pdf_file, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)

    # Loop through each page of the PDF file and extract the text using Tesseract
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        print(text)
