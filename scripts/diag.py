#!/usr/bin/env python3
"""
Detailed debugging script to find exactly where MinerU fails
"""
import os
import sys
import tempfile
import traceback

def debug_mineru_step_by_step():
    """Debug each step of the MinerU processing pipeline"""
    
    pdf_path = '/app/document_store/test_doc_123/original.pdf'
    print(f"🔍 Debugging MinerU pipeline with: {pdf_path}")
    
    try:
        # Step 1: Import all required modules
        print("\n=== Step 1: Imports ===")
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
        print("✅ FileBasedDataWriter, FileBasedDataReader")
        
        from magic_pdf.data.dataset import PymuDocDataset
        print("✅ PymuDocDataset")
        
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        print("✅ doc_analyze")
        
        from magic_pdf.config.enums import SupportedPdfParseMethod
        print("✅ SupportedPdfParseMethod")
        
        import magic_pdf.model as model_config
        print("✅ model_config")
        
        # Step 2: Set model mode
        print("\n=== Step 2: Model Configuration ===")
        model_config.__model_mode__ = 'full'
        print("✅ Set model mode to 'full'")
        
        # Step 3: Create temp directory and writers
        print("\n=== Step 3: Temporary Directory Setup ===")
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"✅ Created temp directory: {temp_dir}")
            
            local_image_dir = os.path.join(temp_dir, "images")
            os.makedirs(local_image_dir, exist_ok=True)
            print(f"✅ Created image directory: {local_image_dir}")
            
            image_writer = FileBasedDataWriter(local_image_dir)
            md_writer = FileBasedDataWriter(temp_dir)
            print("✅ Created FileBasedDataWriter instances")
            
            # Step 4: Read PDF file
            print("\n=== Step 4: PDF File Reading ===")
            reader = FileBasedDataReader("")
            pdf_bytes = reader.read(pdf_path)
            print(f"✅ Read PDF: {len(pdf_bytes)} bytes")
            
            # Step 5: Create PymuDocDataset
            print("\n=== Step 5: PymuDocDataset Creation ===")
            ds = PymuDocDataset(pdf_bytes)
            print("✅ Created PymuDocDataset")
            
            # Step 6: Classify PDF
            print("\n=== Step 6: PDF Classification ===")
            classification = ds.classify()
            print(f"✅ PDF classified as: {classification}")
            
            # Step 7: Apply doc_analyze
            print("\n=== Step 7: Document Analysis ===")
            if classification == SupportedPdfParseMethod.OCR:
                print("📝 Using OCR mode")
                infer_result = ds.apply(doc_analyze, lang='en', ocr=True)
                print("✅ doc_analyze with OCR completed")
                
                print("\n=== Step 8: OCR Pipeline ===")
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
                print("✅ OCR pipeline completed")
                
            else:
                print("📝 Using text mode")
                infer_result = ds.apply(doc_analyze, ocr=False)
                print("✅ doc_analyze without OCR completed")
                
                print("\n=== Step 8: Text Pipeline ===")
                pipe_result = infer_result.pipe_txt_mode(image_writer)
                print("✅ Text pipeline completed")
            
            # Step 9: Generate middle.json
            print("\n=== Step 9: Middle JSON Generation ===")
            name_without_suff = os.path.splitext(os.path.basename(pdf_path))[0]
            middle_json_file = os.path.join(temp_dir, f"{name_without_suff}_middle.json")
            
            try:
                if hasattr(pipe_result, 'dump_middle_json'):
                    pipe_result.dump_middle_json(md_writer, middle_json_file)
                    print("✅ dump_middle_json completed")
                else:
                    print("⚠️  dump_middle_json not available, using fallback")
                    import json
                    pdf_info_to_dump = pipe_result.to_dict().get('pdf_info', []) if hasattr(pipe_result, 'to_dict') else pipe_result.get('pdf_info', [])
                    with open(middle_json_file, 'w', encoding='utf-8') as f:
                        json.dump({'pdf_info': pdf_info_to_dump}, f, indent=4, ensure_ascii=False)
                    print("✅ Fallback middle.json generation completed")
                    
            except Exception as e:
                print(f"⚠️  Middle JSON generation failed: {e}")
                # Continue without middle.json for now
            
            # Step 10: Read and validate middle.json
            print("\n=== Step 10: Middle JSON Reading ===")
            try:
                import json
                with open(middle_json_file, 'r', encoding='utf-8') as f:
                    loaded_middle_data = json.load(f)
                
                pdf_info_list = loaded_middle_data.get('pdf_info', [])
                print(f"✅ Read middle.json: {len(pdf_info_list)} PDF info items")
                
                if pdf_info_list:
                    print("✅ PDF info list is not empty")
                    
                    # Step 11: Process content with bbox
                    print("\n=== Step 11: Content Extraction ===")
                    from mineru_ingester import extract_content_with_bbox
                    
                    image_dir_prefix = "images"
                    content_list_with_bbox = extract_content_with_bbox(pdf_info_list, image_dir_prefix)
                    print(f"✅ Extracted {len(content_list_with_bbox)} content items with bbox")
                    
                    print("\n🎉 SUCCESS: All steps completed successfully!")
                    return True
                    
                else:
                    print("❌ PDF info list is empty")
                    return False
                    
            except Exception as e:
                print(f"❌ Middle JSON reading failed: {e}")
                traceback.print_exc()
                return False
    
    except Exception as e:
        print(f"\n❌ Pipeline failed at current step: {e}")
        print("\n📋 Full traceback:")
        traceback.print_exc()
        return False

def debug_with_original_ingest_pdf():
    """Test the original ingest_pdf function with detailed error capture"""
    
    print("\n" + "="*60)
    print("🧪 Testing Original ingest_pdf Function")
    print("="*60)
    
    pdf_path = '/app/document_store/test_doc_123/original.pdf'
    
    try:
        from mineru_ingester import ingest_pdf
        print(f"📁 Testing ingest_pdf with: {pdf_path}")
        
        result = ingest_pdf(pdf_path, lang='en', dump_intermediate=True)
        
        print(f"📊 Result type: {type(result)}")
        
        if result is None:
            print("❌ ingest_pdf returned None")
        elif isinstance(result, dict):
            content_list = result.get('content_list', [])
            images = result.get('images', {})
            print(f"✅ ingest_pdf succeeded: {len(content_list)} content items, {len(images)} images")
        else:
            print(f"❓ ingest_pdf returned unexpected type: {type(result)}")
            
    except Exception as e:
        print(f"❌ ingest_pdf failed with exception: {e}")
        print("\n📋 Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 Detailed MinerU Debugging")
    print("=" * 60)
    
    # First test step-by-step
    success = debug_mineru_step_by_step()
    
    if success:
        print("\n✅ Step-by-step debugging successful!")
        print("The issue might be in the ingest_pdf function itself.")
    
    # Then test the original function
    debug_with_original_ingest_pdf()