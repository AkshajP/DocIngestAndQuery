#!/usr/bin/env python3
"""
Test script to verify PDF processing works in Docker environment.
Run this inside your Docker container to test the fixes.
"""

import os
import sys
import logging
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_docker_environment():
    """Test Docker environment setup"""
    print("=== Docker Environment Test ===")
    
    # Check if in Docker
    is_docker = os.path.exists('/.dockerenv') or os.getenv('CONTAINER_ENV') == 'docker'
    print(f"Running in Docker: {is_docker}")
    
    # Check environment variables
    env_vars = [
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
        'MINERU_DISABLE_MULTIPROCESSING', 'TOKENIZERS_PARALLELISM'
    ]
    
    for var in env_vars:
        value = os.getenv(var, 'Not set')
        print(f"{var}: {value}")
    
    # Check shared memory
    shm_path = '/dev/shm'
    if os.path.exists(shm_path):
        try:
            # Test writing to shared memory
            test_file = os.path.join(shm_path, 'test_write')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print("✅ Shared memory is writable")
        except Exception as e:
            print(f"❌ Shared memory issue: {e}")
    else:
        print("⚠️  /dev/shm not found")

def test_pdf_processing_minimal():
    """Test PDF processing with minimal setup"""
    print("\n=== Minimal PDF Processing Test ===")
    
    # Test if we can import the Docker-safe version
    try:
        from docker_safe_mineru import ingest_pdf_docker_safe, configure_for_docker
        print("✅ Docker-safe MinerU imported successfully")
        
        # Configure environment
        configure_for_docker()
        print("✅ Environment configured for Docker")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test if we can create temp directories
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"✅ Temporary directory created: {temp_dir}")
            
            # Test file operations
            test_file = os.path.join(temp_dir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            
            with open(test_file, 'r') as f:
                content = f.read()
            
            print(f"✅ File operations work: {content}")
            
    except Exception as e:
        print(f"❌ Temp directory test failed: {e}")
        return False
    
    return True

def test_with_sample_pdf():
    """Test with a sample PDF if available"""
    print("\n=== Sample PDF Test ===")
    
    # Look for PDFs in common locations
    pdf_paths = [
        '/app/document_store/test_doc*/original.pdf',
        '/app/test.pdf',
        '/tmp/test.pdf'
    ]
    
    import glob
    
    found_pdf = None
    for pattern in pdf_paths:
        matches = glob.glob(pattern)
        if matches:
            found_pdf = matches[0]
            break
    
    if not found_pdf:
        print("ℹ️  No sample PDF found - skipping PDF test")
        return True
    
    print(f"Found PDF: {found_pdf}")
    
    try:
        from docker_safe_mineru import ingest_pdf_docker_safe
        
        # Test PDF processing
        result = ingest_pdf_docker_safe(found_pdf, lang='en')
        
        if result and isinstance(result, dict):
            content_list = result.get('content_list', [])
            images = result.get('images', {})
            print(f"✅ PDF processed successfully: {len(content_list)} items, {len(images)} images")
            return True
        else:
            print(f"❌ PDF processing returned invalid result: {result}")
            return False
            
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🐳 Docker PDF Processing Test")
    print("=" * 50)
    
    # Test 1: Docker environment
    test_docker_environment()
    
    # Test 2: Basic imports and setup
    if not test_pdf_processing_minimal():
        print("\n❌ Basic tests failed - check Docker configuration")
        return False
    
    # Test 3: Sample PDF processing
    test_with_sample_pdf()
    
    print("\n✅ All tests completed!")
    print("\nIf any tests failed:")
    print("1. Check Docker resource limits (memory, CPU)")
    print("2. Ensure shared memory is available (/dev/shm)")
    print("3. Verify ML library compatibility")
    print("4. Check for signal handling conflicts")

if __name__ == "__main__":
    main()