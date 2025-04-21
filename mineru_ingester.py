import os
import json
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
import magic_pdf.model as model_config
import tempfile

def ingest_pdf(pdf_file_path, lang= 'en'):
    """
    Process a PDF file and return the extracted JSON data.
    
    Args:
        pdf_file_path (str): Path to the PDF file to be processed
        
    Returns:
        dict: A dictionary containing all extracted data from the PDF:
            - content_list: List of content items extracted from the PDF
            - middle_json: Intermediate JSON representation
            - markdown: Markdown representation of the PDF content
    """
    # Set model mode to full
    model_config.__model_mode__ = 'full'
    
    # Create temporary directories for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        local_image_dir = os.path.join(temp_dir, "images")
        os.makedirs(local_image_dir, exist_ok=True)
        
        # Initialize data writers for the temporary directory
        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(temp_dir)
        
        # Get PDF base name without extension
        name_without_suff = os.path.splitext(os.path.basename(pdf_file_path))[0]
        
        # Read PDF content
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_file_path)
        
        # Create Dataset Instance
        ds = PymuDocDataset(pdf_bytes)
        
        # Process based on PDF classification
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, lang= lang, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)
        
        # Create temporary filenames for data extraction
        content_list_file = os.path.join(temp_dir, f"{name_without_suff}_content_list.json")
        middle_json_file = os.path.join(temp_dir, f"{name_without_suff}_middle.json")
        markdown_file = os.path.join(temp_dir, f"{name_without_suff}.md")
        
        # Using relative path for images in the markdown
        image_dir = "images"
        
        # Dump files to the temporary directory
        pipe_result.dump_content_list(md_writer, content_list_file, image_dir)
        pipe_result.dump_middle_json(md_writer, middle_json_file)
        pipe_result.dump_md(md_writer, markdown_file, image_dir)
        
        # Read the JSON files
        with open(content_list_file, 'r', encoding='utf-8') as f:
            content_list = json.load(f)
        
        with open(middle_json_file, 'r', encoding='utf-8') as f:
            middle_json = json.load(f)
        
        # Read the markdown file
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Collect images if needed (optional)
        images = {}
        for root, _, files in os.walk(local_image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    with open(os.path.join(root, file), 'rb') as img_file:
                        images[file] = img_file.read()
        
        # Return all the extracted data
        return {
            "content_list": content_list,
            # "middle_json": middle_json,
            # "markdown": markdown_content,
            "images": images  # Optional: include binary image data
        }

if __name__ == "__main__":
    pdf_path = "output10.pdf"  # Replace with your PDF path
    result = ingest_pdf(pdf_path)
    
    # Print summary of extracted data
    print(f"Content list items: {len(result['content_list'])}")
    # print(f"Middle JSON pages: {len(result['middle_json'].get('pages', []))}")
    # print(f"Markdown length: {len(result['markdown'])} characters")
    print(f"Images extracted: {len(result['images'])}")