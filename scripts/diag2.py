#!/usr/bin/env python3
"""
Detailed tracing of the ingest_pdf function to find where it fails
"""
import os
import sys
import traceback

def trace_ingest_pdf_with_logging():
    """Add detailed logging to trace where ingest_pdf fails"""
    
    pdf_path = '/app/document_store/test_doc_123/original.pdf'
    
    print(f"🔍 Tracing ingest_pdf function with: {pdf_path}")
    print(f"📁 File exists: {os.path.exists(pdf_path)}")
    print(f"📏 File size: {os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 'N/A'}")
    
    # Patch the ingest_pdf function to add debug logging
    import mineru_ingester
    
    # Store original function
    original_ingest_pdf = mineru_ingester.ingest_pdf
    
    def debug_ingest_pdf(pdf_file_path, lang='en', dump_intermediate=False, output_dir=None, storage_adapter=None):
        """Wrapper with detailed logging"""
        
        print(f"\n🚀 DEBUG: ingest_pdf called with:")
        print(f"   pdf_file_path: {pdf_file_path}")
        print(f"   lang: {lang}")
        print(f"   dump_intermediate: {dump_intermediate}")
        print(f"   output_dir: {output_dir}")
        
        try:
            # Call original function with debug wrapper
            print("\n📝 DEBUG: Setting model mode...")
            import magic_pdf.model as model_config
            model_config.__model_mode__ = 'full'
            print("✅ DEBUG: Model mode set")
            
            print("\n📝 DEBUG: Initializing variables...")
            content_list_with_bbox = []
            images = {}
            loaded_middle_data = None
            print("✅ DEBUG: Variables initialized")
            
            print("\n📝 DEBUG: Creating temporary directory...")
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"✅ DEBUG: Temp dir created: {temp_dir}")
                
                local_image_dir = os.path.join(temp_dir, "images")
                os.makedirs(local_image_dir, exist_ok=True)
                print(f"✅ DEBUG: Image dir created: {local_image_dir}")
                
                print("\n📝 DEBUG: Creating data writers...")
                from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
                image_writer = FileBasedDataWriter(local_image_dir)
                md_writer = FileBasedDataWriter(temp_dir)
                print("✅ DEBUG: Data writers created")
                
                print("\n📝 DEBUG: Reading PDF file...")
                reader = FileBasedDataReader("")
                pdf_bytes = reader.read(pdf_file_path)
                print(f"✅ DEBUG: PDF read: {len(pdf_bytes)} bytes")
                
                print("\n📝 DEBUG: Creating PymuDocDataset...")
                from magic_pdf.data.dataset import PymuDocDataset
                ds = PymuDocDataset(pdf_bytes)
                print("✅ DEBUG: PymuDocDataset created")
                
                print("\n📝 DEBUG: Classifying PDF...")
                from magic_pdf.config.enums import SupportedPdfParseMethod
                classification = ds.classify()
                print(f"✅ DEBUG: PDF classified as: {classification}")
                
                print("\n📝 DEBUG: Starting document analysis...")
                from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
                
                if classification == SupportedPdfParseMethod.OCR:
                    print("📝 DEBUG: Using OCR mode...")
                    infer_result = ds.apply(doc_analyze, lang=lang, ocr=True)
                    print("✅ DEBUG: doc_analyze completed")
                    
                    print("📝 DEBUG: Running OCR pipeline...")
                    pipe_result = infer_result.pipe_ocr_mode(image_writer)
                    print("✅ DEBUG: OCR pipeline completed")
                else:
                    print("📝 DEBUG: Using text mode...")
                    infer_result = ds.apply(doc_analyze, ocr=False)
                    print("✅ DEBUG: doc_analyze completed")
                    
                    print("📝 DEBUG: Running text pipeline...")
                    pipe_result = infer_result.pipe_txt_mode(image_writer)
                    print("✅ DEBUG: Text pipeline completed")
                
                print(f"\n📝 DEBUG: Pipeline result type: {type(pipe_result)}")
                
                # Continue with the rest of the original function...
                name_without_suff = os.path.splitext(os.path.basename(pdf_file_path))[0]
                middle_json_file = os.path.join(temp_dir, f"{name_without_suff}_middle.json")
                
                print(f"📝 DEBUG: Generating middle.json at: {middle_json_file}")
                try:
                    import json
                    if hasattr(pipe_result, 'dump_middle_json'):
                        pipe_result.dump_middle_json(md_writer, middle_json_file)
                    else:
                        pdf_info_to_dump = pipe_result.to_dict().get('pdf_info', []) if hasattr(pipe_result, 'to_dict') else pipe_result.get('pdf_info', [])
                        with open(middle_json_file, 'w', encoding='utf-8') as f:
                            json.dump({'pdf_info': pdf_info_to_dump}, f, indent=4, ensure_ascii=False)
                    print("✅ DEBUG: middle.json generated")
                except Exception as e:
                    print(f"⚠️ DEBUG: middle.json generation failed: {e}")
                
                print(f"📝 DEBUG: Reading middle.json...")
                try:
                    with open(middle_json_file, 'r', encoding='utf-8') as f:
                        loaded_middle_data = json.load(f)
                    
                    pdf_info_list = loaded_middle_data.get('pdf_info', [])
                    print(f"✅ DEBUG: Read middle.json: {len(pdf_info_list)} items")
                    
                    if not pdf_info_list:
                        print("❌ DEBUG: pdf_info_list is empty!")
                        return None
                    
                    print("📝 DEBUG: Extracting content with bbox...")
                    content_list_with_bbox = mineru_ingester.extract_content_with_bbox(pdf_info_list, "images")
                    print(f"✅ DEBUG: Extracted {len(content_list_with_bbox)} content items")
                    
                    print("📝 DEBUG: Collecting images...")
                    for root, _, files in os.walk(local_image_dir):
                        for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                                try:
                                    with open(os.path.join(root, file), 'rb') as img_file:
                                        relative_img_path = os.path.relpath(os.path.join(root, file), local_image_dir)
                                        images[relative_img_path.replace('\\', '/')] = img_file.read()
                                except IOError as e:
                                    print(f"⚠️ DEBUG: Could not read image {file}: {e}")
                    
                    print(f"✅ DEBUG: Collected {len(images)} images")
                    
                    result = {
                        "content_list": content_list_with_bbox,
                        "images": images
                    }
                    
                    print(f"🎉 DEBUG: SUCCESS! Returning result with {len(content_list_with_bbox)} items, {len(images)} images")
                    return result
                    
                except Exception as e:
                    print(f"❌ DEBUG: Error in middle.json processing: {e}")
                    traceback.print_exc()
                    return None
                
        except Exception as e:
            print(f"❌ DEBUG: Error in ingest_pdf: {e}")
            print("📋 DEBUG: Full traceback:")
            traceback.print_exc()
            return None
    
    # Monkey patch the function
    mineru_ingester.ingest_pdf = debug_ingest_pdf
    
    # Now call it
    print("\n🚀 Starting traced ingest_pdf call...")
    try:
        result = mineru_ingester.ingest_pdf(pdf_path, lang='en')
        print(f"\n📊 Final result: {type(result)}")
        if result:
            print(f"✅ SUCCESS: {len(result.get('content_list', []))} items, {len(result.get('images', {}))} images")
        else:
            print("❌ FAILURE: ingest_pdf returned None")
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        traceback.print_exc()
    
    # Restore original function
    mineru_ingester.ingest_pdf = original_ingest_pdf

if __name__ == "__main__":
    trace_ingest_pdf_with_logging()