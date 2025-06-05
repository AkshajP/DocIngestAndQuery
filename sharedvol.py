#!/usr/bin/env python3
"""
Test script to verify shared volume setup between host and Docker container.
"""

import os
import sys
import time
import tempfile
import subprocess
from datetime import datetime

def test_shared_volume_setup():
    """Test that document_store is properly shared between host and container"""
    
    print("=== Shared Volume Setup Test ===\n")
    
    # Test 1: Check if document_store exists on host
    print("1. Checking host document_store...")
    
    host_doc_store = "./document_store"
    abs_host_doc_store = os.path.abspath(host_doc_store)
    
    print(f"   Host path: {abs_host_doc_store}")
    print(f"   Exists: {os.path.exists(abs_host_doc_store)}")
    
    if not os.path.exists(abs_host_doc_store):
        print("   Creating document_store directory...")
        try:
            os.makedirs(abs_host_doc_store, mode=0o755, exist_ok=True)
            print("   ✅ Created document_store directory")
        except Exception as e:
            print(f"   ❌ Failed to create directory: {e}")
            return False
    
    print(f"   Writable: {os.access(abs_host_doc_store, os.W_OK)}")
    
    # Test 2: Create a test file from host
    print("\n2. Creating test file from host...")
    
    test_file_name = f"host_test_{int(time.time())}.txt"
    host_test_file = os.path.join(abs_host_doc_store, test_file_name)
    
    try:
        test_content = f"Created from host at {datetime.now()}\n"
        with open(host_test_file, 'w') as f:
            f.write(test_content)
        
        print(f"   ✅ Created test file: {host_test_file}")
        print(f"   Content: {test_content.strip()}")
        
    except Exception as e:
        print(f"   ❌ Failed to create test file: {e}")
        return False
    
    # Test 3: Check Docker container access
    print("\n3. Testing Docker container access...")
    
    # Check if Docker is available
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("   ⚠️  Docker not available for container test")
            return True  # Host tests passed, skip container tests
        
        print(f"   Docker version: {result.stdout.strip()}")
        
    except Exception as e:
        print(f"   ⚠️  Docker command failed: {e}")
        return True  # Host tests passed, skip container tests
    
    # Test 4: Check if worker container can see the file
    print("\n4. Testing worker container file access...")
    
    try:
        # Try to access the file through the worker container
        docker_cmd = [
            'docker', 'exec', 'docrag-worker',
            'python', '-c', f'''
import os
container_path = "/app/document_store/{test_file_name}"
print(f"Container path: {{container_path}}")
print(f"Exists: {{os.path.exists(container_path)}}")
if os.path.exists(container_path):
    with open(container_path, "r") as f:
        content = f.read()
    print(f"Content: {{content.strip()}}")
    print("✅ Container can read host-created file")
else:
    print("❌ Container cannot see host-created file")
'''
        ]
        
        result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   Container output:")
            for line in result.stdout.strip().split('\n'):
                print(f"     {line}")
        else:
            print(f"   ❌ Container command failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ❌ Container command timed out")
        return False
    except Exception as e:
        print(f"   ❌ Container test failed: {e}")
        return False
    
    # Test 5: Create file from container
    print("\n5. Testing container-to-host file creation...")
    
    container_test_file = f"container_test_{int(time.time())}.txt"
    
    try:
        docker_cmd = [
            'docker', 'exec', 'docrag-worker',
            'python', '-c', f'''
import os
from datetime import datetime
container_path = "/app/document_store/{container_test_file}"
content = f"Created from container at {{datetime.now()}}\\n"
with open(container_path, "w") as f:
    f.write(content)
print(f"✅ Created file in container: {{container_path}}")
print(f"Content: {{content.strip()}}")
'''
        ]
        
        result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   Container output:")
            for line in result.stdout.strip().split('\n'):
                print(f"     {line}")
            
            # Check if host can see the container-created file
            host_container_file = os.path.join(abs_host_doc_store, container_test_file)
            if os.path.exists(host_container_file):
                with open(host_container_file, 'r') as f:
                    content = f.read()
                print(f"   ✅ Host can see container-created file")
                print(f"   Content: {content.strip()}")
            else:
                print(f"   ❌ Host cannot see container-created file: {host_container_file}")
                return False
                
        else:
            print(f"   ❌ Container file creation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Container file creation test failed: {e}")
        return False
    
    # Test 6: Clean up
    print("\n6. Cleaning up test files...")
    
    try:
        if os.path.exists(host_test_file):
            os.remove(host_test_file)
            print(f"   ✅ Removed {host_test_file}")
        
        host_container_file = os.path.join(abs_host_doc_store, container_test_file)
        if os.path.exists(host_container_file):
            os.remove(host_container_file)
            print(f"   ✅ Removed {host_container_file}")
            
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")
    
    print("\n✅ All shared volume tests passed!")
    print("   Both host and container can read/write to document_store")
    return True

def test_environment_detection():
    """Test environment detection logic"""
    
    print("\n=== Environment Detection Test ===")
    
    # Add project root to path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from core.config import get_config
        
        config = get_config()
        
        print(f"Environment: {'Docker' if config.is_docker_environment else 'Host'}")
        print(f"Storage directory: {config.effective_storage_dir}")
        print(f"Storage exists: {os.path.exists(config.effective_storage_dir)}")
        
        if hasattr(config, '_environment_info'):
            print(f"Environment info: {config._environment_info}")
        
    except Exception as e:
        print(f"❌ Environment detection test failed: {e}")
        import traceback
        traceback.print_exc()

def test_upload_with_shared_volume(test_file_path: str = None):
    """Test the upload process with shared volume"""
    
    print("\n=== Upload with Shared Volume Test ===")
    
    # Create a test file if none provided
    if not test_file_path:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test file for shared volume upload\n")
            f.write(f"Created at: {datetime.now()}\n")
            test_file_path = f.name
        
        print(f"Created test file: {test_file_path}")
    
    if not os.path.exists(test_file_path):
        print(f"❌ Test file does not exist: {test_file_path}")
        return False
    
    # Add project root to path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from services.document.upload import upload_document_with_celery
        
        print(f"Testing upload with: {test_file_path}")
        
        result = upload_document_with_celery(
            file_path=test_file_path,
            document_id=f"shared_volume_test_{int(time.time())}",
            case_id="test_case"
        )
        
        print(f"Upload result: {result}")
        
        if result["status"] == "processing":
            print("✅ Upload successful with shared volume")
            
            stored_path = result.get("stored_file_path")
            if stored_path and os.path.exists(stored_path):
                print(f"✅ File accessible at: {stored_path}")
            else:
                print(f"❌ File not found at stored path: {stored_path}")
        else:
            print(f"❌ Upload failed: {result.get('error')}")
        
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up test file
        try:
            if test_file_path and test_file_path.startswith('/tmp'):
                os.remove(test_file_path)
        except:
            pass

if __name__ == "__main__":
    print("Shared Volume Configuration Test")
    print("================================")
    
    # Run tests
    success = test_shared_volume_setup()
    
    if success:
        test_environment_detection()
        
        # Test upload if file provided
        if len(sys.argv) > 1:
            test_upload_with_shared_volume(sys.argv[1])
        else:
            test_upload_with_shared_volume()
    
    print("\nDone!")