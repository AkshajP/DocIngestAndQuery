
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

def is_pdf_file(file_path):
    """Check if the file is a PDF based on its extension"""
    return file_path.lower().endswith('.pdf')

# Function to copy non-PDF files to paginated folder
def copy_non_pdf_file(source_path, tab_name):
    """Copy non-PDF files to the paginated folder"""
    try:
        file_extension = os.path.splitext(source_path)[1]
        destination_path = f"paginated/{tab_name}{file_extension}"
        os.makedirs("paginated", exist_ok=True)
        shutil.copy2(source_path, destination_path)
        print(f"Copied non-PDF file: {tab_name}{file_extension}")
        return True
    except Exception as e:
        print(f"Error copying file {source_path}: {str(e)}")
        return False

# Optimized function to create header content in memory
def create_header_content_in_memory(page_num, page_size, rotation, tab, num_pages):
    """Create header PDF content in memory without file I/O"""
    # Create PDF in memory using BytesIO
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=page_size)
    c.setFont('Helvetica-Bold', 18)
    c.setFillColor(blue)
    
    width, height = page_size
    
    # Save state and apply transformations
    c.saveState()
    if rotation == 0:
        c.translate(int(width/2), int(height) - 30)
    elif rotation == 90:
        c.translate(30, int(height/2))
        c.rotate(90)
    elif rotation == 180:
        c.translate(int(width/2), 30)
        c.rotate(180)
    elif rotation == 270:
        c.translate(int(width)-30, int(height/2))
        c.rotate(270)
    
    # Draw the header text
    c.drawCentredString(0, 0, f"{tab}/{str(page_num).zfill(len(str(num_pages)))}")
    c.restoreState()
    c.save()
    
    # Get the PDF content from memory
    buffer.seek(0)
    return buffer

# Highly optimized function to paginate PDF files
def paginate_pdfs_optimized(file, output_filename, page_number, tab):
    """Optimized PDF pagination with in-memory operations"""
    try:
        # Read input PDF once
        with open(file, "rb") as f:
            input_pdf = PdfReader(f)
            output_pdf = PdfWriter()
            num_pages = len(input_pdf.pages)
            
            # Pre-calculate page info to avoid repeated calculations
            page_info = []
            for i in range(num_pages):
                page = input_pdf.pages[i]
                page_media_box = page.mediabox
                page_size = (float(page_media_box.width), float(page_media_box.height))
                rotation = page.get('/Rotate', 0) or 0
                page_info.append((page_size, rotation))
            
            # Process all pages
            for i in range(num_pages):
                page_number += 1
                page_size, rotation = page_info[i]
                
                # Create header in memory
                header_buffer = create_header_content_in_memory(
                    page_number, page_size, rotation, tab, num_pages
                )
                
                # Read header PDF from memory
                header_pdf = PdfReader(header_buffer)
                
                # Get the page and merge with header
                page = input_pdf.pages[i]
                page.merge_page(header_pdf.pages[0])
                output_pdf.add_page(page)
                
                # Clean up buffer
                header_buffer.close()
        
        # Write output file once at the end
        with open(output_filename, "wb") as f:
            output_pdf.write(f)
            
        return page_number
        
    except Exception as e:
        print(f"Error processing PDF {file}: {str(e)}")
        return page_number

# Further optimized version with batch header creation
def paginate_pdfs_ultra_optimized(file, output_filename, page_number, tab):
    """Ultra-optimized version that creates all headers in one pass"""
    try:
        with open(file, "rb") as f:
            input_pdf = PdfReader(f)
            output_pdf = PdfWriter()
            num_pages = len(input_pdf.pages)
            
            # Create all headers in a single PDF document
            headers_buffer = BytesIO()
            headers_canvas = canvas.Canvas(headers_buffer, pagesize=A4)
            
            # Store page information and header page indices
            page_info = []
            header_page_count = 0
            
            for i in range(num_pages):
                page = input_pdf.pages[i]
                page_media_box = page.mediabox
                page_size = (float(page_media_box.width), float(page_media_box.height))
                rotation = page.get('/Rotate', 0) or 0
                
                # Create new page in headers PDF with correct size
                headers_canvas.setPageSize(page_size)
                headers_canvas.setFont('Helvetica-Bold', 18)
                headers_canvas.setFillColor(blue)
                
                width, height = page_size
                current_page_num = page_number + i + 1
                
                # Apply transformations and draw text
                headers_canvas.saveState()
                if rotation == 0:
                    headers_canvas.translate(int(width/2), int(height) - 30)
                elif rotation == 90:
                    headers_canvas.translate(30, int(height/2))
                    headers_canvas.rotate(90)
                elif rotation == 180:
                    headers_canvas.translate(int(width/2), 30)
                    headers_canvas.rotate(180)
                elif rotation == 270:
                    headers_canvas.translate(int(width)-30, int(height/2))
                    headers_canvas.rotate(270)
                
                headers_canvas.drawCentredString(0, 0, f"{tab}/{str(current_page_num).zfill(len(str(num_pages)))}")
                headers_canvas.restoreState()
                headers_canvas.showPage()  # Move to next page
                
                page_info.append((page_size, rotation, header_page_count))
                header_page_count += 1
            
            # Finalize headers PDF
            headers_canvas.save()
            headers_buffer.seek(0)
            headers_pdf = PdfReader(headers_buffer)
            
            # Now merge all pages with their corresponding headers
            for i in range(num_pages):
                page = input_pdf.pages[i]
                _, _, header_idx = page_info[i]
                
                # Merge with corresponding header page
                page.merge_page(headers_pdf.pages[header_idx])
                output_pdf.add_page(page)
            
            # Clean up
            headers_buffer.close()
        
        # Write output file
        with open(output_filename, "wb") as f:
            output_pdf.write(f)
            
        return page_number + num_pages
        
    except Exception as e:
        print(f"Error processing PDF {file}: {str(e)}")
        return page_number

# Read the Excel file containing file URLs and page numbers
def getIndexValue(headers, columnName):
    try:
        return headers.index(columnName)
    except ValueError as e:
        raise ValueError(f"Header not found in the first row: {str(e)}")

# Main processing function with optimizations
def process_files_optimized(data, headers, hostNumber):
    """Optimized main processing function"""
    # Get column indices once
    local_file_path_index = getIndexValue(headers, 'Local_Path')
    pagination_status_index = getIndexValue(headers, 'pagination_status')
    tab_name_index = getIndexValue(headers, 'Tab')
    pagination_host_index = getIndexValue(headers, 'host_no')
    
    # Ensure paginated directory exists
    os.makedirs("paginated", exist_ok=True)
    
    # Pre-filter rows to process
    rows_to_process = []
    for i, row in enumerate(data[1:], 1):
        status = row[pagination_status_index]
        host = row[pagination_host_index]
        if status == 'PENDING' and int(host) == hostNumber:
            rows_to_process.append((i, row))
    
    print(f"Processing {len(rows_to_process)} files...")
    
    # Process filtered rows
    for rowNum, row in rows_to_process:
        name = row[tab_name_index]
        local_file_location = row[local_file_path_index]
        
        # Check if file exists
        if not os.path.exists(local_file_location):
            print(f"File not found: {local_file_location}")
            continue
        
        # Process based on file type
        if is_pdf_file(local_file_location):
            # Use the ultra-optimized version for best performance
            page_number = paginate_pdfs_ultra_optimized(
                local_file_location, 
                f"paginated/{name}.pdf", 
                0, 
                name
            )
            if page_number:
                print(f"Row {rowNum+1}, Status: DONE, Pages: {page_number}")
                print(f"Processed PDF: {name}")
        else:
            # Copy non-PDF files
            if copy_non_pdf_file(local_file_location, name):
                print(f"Row {rowNum+1}, Status: DONE (COPIED)")
                print(f"Non-PDF file copied: {name}")
            else:
                print(f"Failed to copy non-PDF file: {name}")
    
    print("Done")

# Example usage (replace with your actual data and headers):
process_files_optimized(data, headers, hostNumber)
print("Done")
