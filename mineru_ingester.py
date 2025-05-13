import os
import json
import re
import tempfile

# --- Imports from magic_pdf (Ensure these are correct for your setup) ---
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
import magic_pdf.model as model_config
from magic_pdf.config.ocr_content_type import BlockType, ContentType
from magic_pdf.post_proc.para_split_v3 import ListLineTag

# --- Helper Functions (Adapted from middle_json_processor_v1) ---

def join_path(base, *paths):
    """Simple simulation of os.path.join or urljoin for paths/URLs."""
    if not base: # If base is empty, just join paths normally
        return os.path.join(*paths)
    # Basic check if it looks like a URL
    if base.startswith(('http://', 'https://', 's3://')):
        if base.endswith('/'):
            base = base[:-1]
        parts = [base] + [str(p).strip('/') for p in paths]
        return "/".join(parts)
    else: # Assume local path
        return os.path.join(base, *paths)

def __is_hyphen_at_line_end(line):
    """Check if a line ends with one or more letters followed by a hyphen."""
    return bool(re.search(r'[A-Za-z]+-\s*$', line))

def detect_lang(text):
    """Simplified language detection (checks proportion of Latin chars)."""
    if not text or not isinstance(text, str):
        return 'empty'
    en_pattern = r'[a-zA-Z]+'
    en_matches = re.findall(en_pattern, text)
    en_length = sum(len(match) for match in en_matches)
    total_len = len(text)
    if total_len > 0:
        if en_length / total_len >= 0.5:
            return 'en'
        else:
            if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text):
                return 'zh'
            return 'unknown'
    else:
        return 'empty'

def full_to_half(text: str) -> str:
    """Convert full-width characters to half-width characters."""
    result = []
    if not isinstance(text, str):
        return text
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E and code != 0xFF3C:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            result.append(' ')
        else:
            result.append(char)
    return ''.join(result)

def merge_para_with_text(para_block):
    """Merges lines and spans within a block into a single text string, handling spacing."""
    block_text_for_lang_detect = ''
    lines_data = para_block.get('lines', [])

    for line in lines_data:
        for span in line.get('spans', []):
            if span.get('type') == ContentType.Text:
                content = span.get('content', '')
                block_text_for_lang_detect += full_to_half(content or '')

    block_lang = detect_lang(block_text_for_lang_detect)

    para_text = ''
    for i, line in enumerate(lines_data):
        if i >= 1 and line.get(ListLineTag.IS_LIST_START_LINE, False):
            para_text += '  \n'

        line_content_list = []
        for j, span in enumerate(line.get('spans', [])):
            span_type = span.get('type')
            content = span.get('content', '')

            if span_type == ContentType.Text:
                content = full_to_half(content or '')
                # content = ocr_escape_special_markdown_char(content) # Removed markdown escaping
            elif span_type == ContentType.InlineEquation:
                content = f"${content}$"
            elif span_type == ContentType.InterlineEquation:
                content = f"\n$$\n{content}\n$$\n"
            else:
                content = ''

            content = content.strip()

            if content:
                line_content_list.append({
                    "content": content,
                    "is_last_span": j == len(line.get('spans', [])) - 1,
                    "type": span_type
                })

        for k, span_data in enumerate(line_content_list):
            content = span_data["content"]
            para_text += content

            is_last_span_in_line = k == len(line_content_list) - 1

            if not is_last_span_in_line:
                if block_lang not in ['zh', 'ja', 'ko']:
                    if not (span_data["type"] == ContentType.Text and __is_hyphen_at_line_end(content)):
                        para_text += ' '
                else:
                    if span_data["type"] != ContentType.InlineEquation:
                        para_text += ' '
                    else:
                        para_text += ' '

        if i < len(lines_data) - 1:
            last_span_content = line_content_list[-1]["content"] if line_content_list else ""
            last_span_type = line_content_list[-1]["type"] if line_content_list else None

            if block_lang in ['zh', 'ja', 'ko']:
                if last_span_type == ContentType.InlineEquation:
                    para_text += ' '
            else:
                if not (last_span_type == ContentType.Text and __is_hyphen_at_line_end(last_span_content)):
                    para_text += ' '
                else:
                    if para_text.endswith('-'):
                        para_text = para_text[:-1]

    return para_text.strip()

def get_title_level(block):
    """Extracts title level, ensuring it's within a reasonable range (e.g., 1-6)."""
    title_level = block.get('level', 0)

    if title_level <= 0:
        return 0
    elif title_level > 6:
        return 6
    else:
        return int(title_level)

# --- Main Processing Function (Adapted from middle_json_processor_v1) ---
def extract_content_with_bbox(pdf_info_list: list, img_buket_path: str = ''):
    """
    Processes the pdf_info list (from magic-pdf result) to generate a content list
    including the bbox for each item.
    """
    output_content = []

    for page_info in pdf_info_list:
        page_idx = page_info.get('page_idx', -1)
        paras_of_layout = page_info.get('preproc_blocks', [])

        if not paras_of_layout:
            continue

        for para_block in paras_of_layout:
            para_type = para_block.get('type')
            para_content = {}
            block_bbox = para_block.get('bbox')

            # --- Determine content based on block type ---
            if para_type in [BlockType.Text, BlockType.List, BlockType.Index]:
                merged_text = merge_para_with_text(para_block)
                if merged_text:
                    para_content = {'type': 'text', 'text': merged_text}
            elif para_type == BlockType.Title:
                merged_text = merge_para_with_text(para_block)
                if merged_text:
                    para_content = {'type': 'text', 'text': merged_text}
                    title_level = get_title_level(para_block)
                    if title_level > 0:
                        para_content['text_level'] = title_level
            elif para_type == BlockType.InterlineEquation:
                merged_text = merge_para_with_text(para_block)
                if merged_text:
                    para_content = {
                        'type': 'equation',
                        'text': merged_text.strip('$'),
                        'text_format': 'latex',
                    }
            elif para_type == BlockType.Image:
                para_content = {'type': 'image', 'img_path': '', 'img_caption': [], 'img_footnote': []}
                nested_blocks = para_block.get('blocks', [para_block]) # Process self if no nested blocks

                for block in nested_blocks:
                    block_type_inner = block.get('type')
                    if block_type_inner == BlockType.ImageBody or block_type_inner == BlockType.Image:
                        for line in block.get('lines', []):
                            for span in line.get('spans', []):
                                if span.get('type') == ContentType.Image and span.get('image_path'):
                                    para_content['img_path'] = join_path(img_buket_path, span['image_path'])
                                    break
                            if para_content['img_path']: break
                    elif block_type_inner == BlockType.ImageCaption:
                        caption_text = merge_para_with_text(block)
                        if caption_text: para_content['img_caption'].append(caption_text)
                    elif block_type_inner == BlockType.ImageFootnote:
                        footnote_text = merge_para_with_text(block)
                        if footnote_text: para_content['img_footnote'].append(footnote_text)

                if not para_content.get('img_path'): para_content = {}

            elif para_type == BlockType.Table:
                para_content = {'type': 'table', 'table_body': '', 'img_path': '', 'table_caption': [], 'table_footnote': []}
                nested_blocks = para_block.get('blocks', [para_block])

                for block in nested_blocks:
                    block_type_inner = block.get('type')
                    if block_type_inner == BlockType.TableBody or block_type_inner == BlockType.Table:
                        for line in block.get('lines', []):
                            for span in line.get('spans', []):
                                if span.get('type') == ContentType.Table:
                                    if span.get('latex'):
                                        para_content['table_body'] = span['latex']
                                        para_content['table_format'] = 'latex'
                                        break
                                    elif span.get('html'):
                                        para_content['table_body'] = span['html']
                                        para_content['table_format'] = 'html'
                                        break
                                    elif span.get('image_path'):
                                        para_content['img_path'] = join_path(img_buket_path, span['image_path'])
                                        para_content['type'] = 'image'
                                        para_content.pop('table_body', None)
                                        break
                            if para_content.get('table_body') or para_content.get('img_path'): break
                    elif block_type_inner == BlockType.TableCaption:
                        caption_text = merge_para_with_text(block)
                        if caption_text: para_content['table_caption'].append(caption_text)
                    elif block_type_inner == BlockType.TableFootnote:
                        footnote_text = merge_para_with_text(block)
                        if footnote_text: para_content['table_footnote'].append(footnote_text)

                if not para_content.get('table_body') and not para_content.get('img_path'):
                    para_content = {}

            # --- Finalize and Append ---
            if para_content:
                para_content['page_idx'] = page_idx
                if block_bbox:
                    para_content['bbox'] = block_bbox
                output_content.append(para_content)

    return output_content


# --- Updated PDF Ingestion Function ---
def ingest_pdf(pdf_file_path, lang='en', dump_intermediate=False, output_dir=None, storage_adapter=None):
    """
    Processes a PDF file using magic-pdf and returns extracted content
    with bounding boxes and associated images.

    Args:
        pdf_file_path (str): Path to the PDF file.
        lang (str): Language hint for OCR ('en', 'zh', etc.). Defaults to 'en'.
        dump_intermediate (bool): If True, saves intermediate middle.json and .md files
                                    in the temporary directory. Defaults to False.
        output_dir (str): Directory where to save extracted images. If None, images won't be saved.
        storage_adapter: Optional storage adapter to use for saving files. If None, uses direct file I/O.

    Returns:
        dict: A dictionary containing:
            - "content": List of content items (dict), each including text/data
                         and a 'bbox' key.
            - "images": Dictionary where keys are image filenames and values are
                      image binary data. Returns None if image extraction fails.
        Returns None if PDF processing fails at an early stage.
    """
    # Set model mode to full (ensure this is appropriate for your use case)
    model_config.__model_mode__ = 'full'

    content_list_with_bbox = []
    images = {}
    loaded_middle_data = None # Variable to store data read from middle.json

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_image_dir = os.path.join(temp_dir, "images")
            os.makedirs(local_image_dir, exist_ok=True)

            # Initialize data writers (needed for image extraction by magic-pdf)
            image_writer = FileBasedDataWriter(local_image_dir)
            md_writer = FileBasedDataWriter(temp_dir) # Potentially needed for md dump

            name_without_suff = os.path.splitext(os.path.basename(pdf_file_path))[0]

            # Read PDF content
            reader = FileBasedDataReader("")
            pdf_bytes = reader.read(pdf_file_path)

            # Create Dataset Instance
            ds = PymuDocDataset(pdf_bytes)

            # Process based on PDF classification
            if ds.classify() == SupportedPdfParseMethod.OCR:
                infer_result = ds.apply(doc_analyze, lang=lang, ocr=True)
                # pipe_ocr_mode likely returns the structured data including pdf_info
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)
                # pipe_txt_mode likely returns the structured data including pdf_info
                pipe_result = infer_result.pipe_txt_mode(image_writer)

            # --- Generate middle.json ---
            middle_json_file = os.path.join(temp_dir, f"{name_without_suff}_middle.json")
            try:
                if hasattr(pipe_result, 'dump_middle_json'):
                    pipe_result.dump_middle_json(md_writer, middle_json_file)
                else:
                    # Fallback to manual dump if dump_middle_json is not available
                    pdf_info_to_dump = pipe_result.to_dict().get('pdf_info', []) if hasattr(pipe_result, 'to_dict') else pipe_result.get('pdf_info', [])
                    with open(middle_json_file, 'w', encoding='utf-8') as f:
                        json.dump({'pdf_info': pdf_info_to_dump}, f, indent=4, ensure_ascii=False)
                print(f"Generated middle.json at: {middle_json_file}")
            except Exception as e:
                print(f"Warning: Failed to generate middle.json - {e}")

            # --- Read middle.json into a variable ---
            try:
                with open(middle_json_file, 'r', encoding='utf-8') as f:
                    loaded_middle_data = json.load(f)
                print(f"Read middle.json from: {middle_json_file}")
            except Exception as e:
                print(f"Error reading middle.json: {e}")
                return None

            # --- Extract pdf_info_list from loaded data ---
            pdf_info_list = loaded_middle_data.get('pdf_info', [])
            if not pdf_info_list:
                print("Warning: 'pdf_info' not found in loaded middle.json.")
                return None

            # --- Process pdf_info_list to get content with bbox ---
            # Use a relative path prefix matching where magic-pdf saves images
            image_dir_prefix = "images"
            content_list_with_bbox = extract_content_with_bbox(pdf_info_list, image_dir_prefix)

            # --- Dump markdown (Optional) ---
            if dump_intermediate:
                markdown_file = os.path.join(temp_dir, f"{name_without_suff}.md")
                try:
                    # Use infer_result for markdown dump
                    if hasattr(infer_result, 'dump_md'):
                        infer_result.dump_md(md_writer, markdown_file, image_dir_prefix)
                except Exception as e:
                    print(f"Warning: Failed to dump markdown - {e}")

            # --- Collect images ---
            # Images should have been written to local_image_dir by magic-pdf's image_writer
            for root, _, files in os.walk(local_image_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        try:
                            with open(os.path.join(root, file), 'rb') as img_file:
                                # Use the relative path within 'images' dir as the key
                                relative_img_path = os.path.relpath(os.path.join(root, file), local_image_dir)
                                # Normalize path separators for consistency if needed
                                images[relative_img_path.replace('\\', '/')] = img_file.read()
                        except IOError as e:
                            print(f"Warning: Could not read image file {file}: {e}")

    except ImportError as e:
        print(f"ImportError: Please ensure magic-pdf is installed correctly. {e}")
        return None
    except FileNotFoundError:
        print(f"Error: Input PDF file not found at {pdf_file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during PDF processing for {pdf_file_path}: {e}")
        # Depending on the error, you might still have partial results
        # For safety, return None on major errors during processing
        return None

    # --- Save images to the specified output directory if provided ---
    if images and output_dir:
        try:
            # Create the output directory if needed
            if storage_adapter:
                storage_adapter.create_directory(output_dir)
            else:
                os.makedirs(output_dir, exist_ok=True)
                
            for filename, image_data in images.items():
                filepath = os.path.join(output_dir, filename)
                try:
                    # Use storage adapter if provided, otherwise use direct file I/O
                    if storage_adapter:
                        storage_adapter.write_file(image_data, filepath)
                    else:
                        # Create subdirectories if needed
                        if not os.path.exists(os.path.dirname(filepath)):
                            os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        # Write the image file
                        with open(filepath, 'wb') as f:
                            f.write(image_data)
                    print(f"Saved image to: {filepath}")
                except Exception as e:
                    print(f"Error saving image '{filename}': {e}")
        except Exception as e:
            print(f"Error creating output directory '{output_dir}': {e}")

    # --- Return the final structure ---
    return {
        "content_list": content_list_with_bbox,  
        "images": images
    }
# --- Example Usage ---
if __name__ == "__main__":
    # pdf_path = "output10.pdf"  # Replace with your PDF path
    pdf_path = "/Users/vikas/Downloads/docllm/output10.pdf" # Example path

    if not os.path.exists(pdf_path):
        print(f"Error: The example PDF '{pdf_path}' does not exist. Please update the path.")
    else:
        # Set dump_intermediate=True if you want to see the markdown file as well
        result = ingest_pdf(pdf_path, dump_intermediate=True)

        if result:
            print(f"--- Processing Summary for {pdf_path} ---")
            print(f"Content items extracted: {len(result['content'])}")

            if result['images']:
                print(f"Images extracted and saved to the 'images' directory: {len(result['images'])}")
                # print("Image keys:", result['images'][:5]) # Print first 5 image keys
            else:
                print("Images extracted: 0")

            # Optional: Print the first few content items
            # print("\n--- First 3 Content Items (with bbox): ---")
            # for item in result['content'][:3]:
            #    print(json.dumps(item, indent=2, ensure_ascii=False))

            # Optional: Save the final result to a JSON file
            output_final_path = "final_output_with_bbox.json"
            try:
                # Save only the content and the list of image filenames
                with open(output_final_path, 'w', encoding='utf-8') as f:
                    json.dump({"content": result['content'], "images": result['images']}, f, indent=4, ensure_ascii=False)
                print(f"\nContent data (with bbox) and image filenames saved to {output_final_path}")
            except Exception as e:
                print(f"\nError saving content data to JSON: {e}")
        else:
            print(f"Processing failed for {pdf_path}.")