
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import blue
from PyPDF2 import PdfReader, PdfWriter
import os
import requests
from io import BytesIO
from decimal import Decimal

import gspread

import logging
import shutil  # Added for copying files

# C:\teres_\ELM TRADING & CONTRACTING WLL Vs LA JOLLA HOTEL - 27-05-2025\[A] Claimant Submissions\2023.08.10 - Request for Arbitration - ELM v La Jolla.pdf

sheet_name="TERES INDEX"
hostNumber = 25


gc = gspread.service_account()

sh = gc.open_by_key("1IZMf8g807R2GjzipOrk7pDwHqt9wG8AqBJxDJqMgOZs")

worksheet = sh.worksheet(sheet_name)

headers = worksheet.row_values(1)  # Assuming the first row contains headers
data = worksheet.get_all_values()
# Function to check if file is a PDF
def is_pdf_file(file_path):
    """Check if the file is a PDF based on its extension"""
    return file_path.lower().endswith('.pdf')

# Function to copy non-PDF files to paginated folder
def copy_non_pdf_file(source_path, tab_name):
    """Copy non-PDF files to the paginated folder"""
    try:
        # Get the file extension
        file_extension = os.path.splitext(source_path)[1]
        
        # Create destination path
        destination_path = f"paginated/{tab_name}{file_extension}"
        
        # Ensure the paginated directory exists
        os.makedirs("paginated", exist_ok=True)
        
        # Copy the file
        shutil.copy2(source_path, destination_path)
        
        print(f"Copied non-PDF file: {tab_name}{file_extension}")
        return True
    except Exception as e:
        print(f"Error copying file {source_path}: {str(e)}")
        return False

# Function to create a PDF with a header
def create_header_pdf(page_num, page_size,rotation,tab,num_pages):
    c = canvas.Canvas(f"header_{str(page_num)}.pdf", pagesize=page_size)
    c.setFont('Helvetica-Bold', 18)
    c.setFillColor(blue)
    width, height = page_size
    # c.drawCentredString(int(width)-100, int(height) - 40, f"{tab}{str(page_num)}")
    
    c.saveState();
    if rotation == 0:
        # c.translate(int(width)-100, int(height) - 30)
        c.translate(int(int(width)/2), int(height) - 30)
    elif rotation == 90:
        # c.translate(30, int(height)-100)
        c.translate(30, int(int(height)/2))
        c.rotate(90)
    elif rotation == 180:
        # c.translate(100, 30)
        c.translate(int(int(width)/2), 30)
        c.rotate(180)
    elif rotation == 270:
        # c.translate(int(width)-30, 100)
        c.translate(int(width)-30, int(int(height)/2))
        c.rotate(270)

    c.drawCentredString(0,0, f"{tab}/{str(page_num).zfill(len(str(num_pages)))}")
    c.save()

    return f"header_{str(page_num)}.pdf"

# Function to paginate and merge PDF files with a cover and index
def paginate_pdfs(file, output_filename, page_number,tab):
    output_pdf = PdfWriter()
    

    input_pdf = PdfReader(open(file, "rb"))
    
    num_pages = len(input_pdf.pages)
    for i in range(num_pages):
        page_number += 1

        # Create header page with the page number
        page = input_pdf.pages[i]
        page_media_box = page.mediabox
        page_size = (page_media_box.width, page_media_box.height)
        rotation = page.get('/Rotate', 0)

        header_pdf_filename = create_header_pdf(page_number, page_size,rotation,tab,num_pages)
        header_pdf = PdfReader(open(header_pdf_filename, "rb"))
        
        # Get the actual page and merge with the header
        page = input_pdf.pages[i]
        page.merge_page(header_pdf.pages[0])
        
        # Add the merged page to the output
        output_pdf.add_page(page)
        
        # Clean up the header PDF file
        os.remove(header_pdf_filename)

    # Write the output PDF file
    with open(output_filename, "wb") as f:
        output_pdf.write(f)

    
    return page_number


# Read the Excel file containing file URLs and page numbers
def getIndexValue(headers,columnName):
    try:
        s3_col_index = headers.index(columnName)# Column index for S3 file paths
        return s3_col_index  # Column index for status
    except ValueError as e:
        raise ValueError(f"Header not found in the first row: {str(e)}")

local_file_path_index=getIndexValue(headers,'Local_Path')
pagination_status_index=getIndexValue(headers,'pagination_status')
tab_name_index=getIndexValue(headers,'Tab')
pagination_host_index=getIndexValue(headers,'host_no')
# pagination_start_page_index=getIndexValue(headers,'start_page_num')

# print(local_file_path_index,pagination_status_index,start_page_numIndex,tab_name_index)

# Ensure paginated directory exists
os.makedirs("paginated", exist_ok=True)

rowNum=1

for row in data[1:]:
    status=row[pagination_status_index]
    host=row[pagination_host_index]
    if status=='PENDING' and int(host)==hostNumber:

        name=row[tab_name_index]
        local_file_location=row[local_file_path_index]
        
        # Check if file exists
        if not os.path.exists(local_file_location):
            print(f"File not found: {local_file_location}")
            rowNum += 1
            continue

        # Check if the file is a PDF
        if is_pdf_file(local_file_location):
            # Process PDF files with pagination
            # page_number=int(row[pagination_start_page_index])-1
            page_number=0
            page_number = paginate_pdfs(local_file_location,"paginated/"+name+".pdf", page_number,name)
            if(page_number): 
                # worksheet.update_cell(rowNum+1, pagination_status_index+1, 'DONE')
                #worksheet.update_cell(rowNum+1, paginated_pathIndex+1, output_file_path)
                print(rowNum+1, pagination_status_index+1, 'DONE')
                print(name,page_number)
        else:
            # Copy non-PDF files to paginated folder
            if copy_non_pdf_file(local_file_location, name):
                print(rowNum+1, pagination_status_index+1, 'DONE (COPIED)')
                print(f"Non-PDF file copied: {name}")
            else:
                print(f"Failed to copy non-PDF file: {name}")
                
    rowNum=rowNum+1
    
print("Done")