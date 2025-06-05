#!/usr/bin/env python3
"""
Diagnostic script to test PDF processing and identify issues.
Run this to debug PDF extraction problems.
"""
import os
import sys
import traceback
from pathlib import Path

def test_pdf_file(pdf_path):
    """Test basic PDF file properties"""
    print(f"=== PDF File Diagnostics ===")
    print(f"File path: {pdf_path}")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print("❌ File does not exist")
        return False
    print("✅ File exists")
    
    # Check file size
    file_size = os.path.getsize(pdf_path)
    print(f"✅ File size: {file_size:,} bytes")
    
    if file_size == 0:
        print("❌ File is empty")
        return False
    
    # Check if readable
    try:
        with open(pdf_path, 'rb') as f:
            header = f.read(10)
        print(f"✅ File is readable, header: {header}")
        
        # Check PDF header
        if header.startswith(b'%PDF-'):
            print("✅ Valid PDF header detected")
        else:
            print("❌ Invalid PDF header")
            return False
            
    except Exception as e:
        print(f"❌ Cannot read file: {e}")
        return False
    
    return True

def test_magic_pdf_imports():
    """Test magic_pdf imports"""
    print(f"\n=== Magic PDF Import Test ===")
    
    try:
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
        print("✅ FileBasedDataWriter/Reader imported")
    except ImportError as e:
        print(f"❌ FileBasedDataWriter/Reader import failed: {e}")
        return False
    
    try:
        from magic_pdf.data.dataset import PymuDocDataset
        print("✅ PymuDocDataset imported")
    except ImportError as e:
        print(f"❌ PymuDocDataset import failed: {e}")
        return False
    
    try:
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        print("✅ doc_analyze imported")
    except ImportError as e:
        print(f"❌ doc_analyze import failed: {e}")
        return False
    
    try:
        from magic_pdf.config.enums import SupportedPdfParseMethod
        print("✅ SupportedPdfParseMethod imported")
    except ImportError as e:
        print(f"❌ SupportedPdfParseMethod import failed: {e}")
        return False
    
    try:
        import magic_pdf.model as model_config
        print("✅ model_config imported")
    except ImportError as e:
        print(f"❌ model_config import failed: {e}")
        return False
    
    return True

def test_mineru_ingester(pdf_path):
    """Test the mineru_ingester function"""
    print(f"\n=== MinerU Ingester Test ===")
    
    try:
        # Import the ingest_pdf function
        sys.path.insert(0, '.')
        from mineru_ingester import ingest_pdf
        print("✅ ingest_pdf imported successfully")
        
        # Test with detailed error capture
        print(f"Processing PDF: {pdf_path}")
        result = ingest_pdf(pdf_path, lang='en', dump_intermediate=True)
        
        print(f"Result type: {type(result)}")
        
        if result is None:
            print("❌ ingest_pdf returned None")
            return False
        
        if isinstance(result, dict):
            content_list = result.get("content_list", [])
            images = result.get("images", {})
            print(f"✅ Content items: {len(content_list)}")
            print(f"✅ Images: {len(images)}")
            
            if len(content_list) == 0:
                print("⚠️  No content extracted")
            else:
                print("✅ Content extraction successful")
                
        return True
        
    except Exception as e:
        print(f"❌ ingest_pdf failed: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def main():
    """Main diagnostic function"""
    if len(sys.argv) != 2:
        print("Usage: python pdf_diagnostic.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    print("🔍 PDF Processing Diagnostic Tool")
    print("=" * 50)
    
    # Test 1: PDF file basics
    if not test_pdf_file(pdf_path):
        print("\n❌ PDF file test failed - cannot proceed")
        return
    
    # Test 2: Magic PDF imports
    if not test_magic_pdf_imports():
        print("\n❌ Magic PDF import test failed - check installation")
        return
    
    # Test 3: MinerU ingester
    if not test_mineru_ingester(pdf_path):
        print("\n❌ MinerU ingester test failed")
        return
    
    print("\n✅ All tests passed! PDF processing should work.")

if __name__ == "__main__":
    main()