import pytesseract
import PyPDF2

# Set the path to the PDF file
pdf_file = 'example.pdf'

# Open the PDF file using PyPDF2
with open(pdf_file, 'rb') as file:
    pdf_reader = PyPDF2.PdfFileReader(file)

    # Loop through each page of the PDF file and extract the text using Tesseract
    for page_num in range(pdf_reader.getNumPages()):
        page = pdf_reader.getPage(page_num)
        text = page.extractText()
        print(text)
