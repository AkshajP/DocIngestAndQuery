#!/usr/bin/env python3
"""
Diagnostic utility to debug document registration issues.
"""

import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def diagnose_document_registration(document_id: str = None):
    """Diagnose document registration issues"""
    
    print("=== Document Registration Diagnostics ===")
    
    try:
        # 1. Check DocumentMetadataRepository
        print("\n1. DocumentMetadataRepository Diagnostics:")
        from db.document_store.repository import DocumentMetadataRepository
        
        doc_repo = DocumentMetadataRepository()
        print(f"   Registry path: {doc_repo.storage_path}")
        print(f"   Registry file exists: {os.path.exists(doc_repo.storage_path)}")
        
        if os.path.exists(doc_repo.storage_path):
            print(f"   Registry file size: {os.path.getsize(doc_repo.storage_path)} bytes")
            print(f"   Registry file readable: {os.access(doc_repo.storage_path, os.R_OK)}")
            print(f"   Registry file writable: {os.access(doc_repo.storage_path, os.W_OK)}")
        
        # Check registry contents
        all_documents = doc_repo.list_documents()
        print(f"   Total documents in registry: {len(all_documents)}")
        
        if document_id:
            specific_doc = doc_repo.get_document(document_id)
            print(f"   Document '{document_id}' found: {specific_doc is not None}")
            if specific_doc:
                print(f"   Document status: {specific_doc.get('status')}")
                print(f"   Document case_id: {specific_doc.get('case_id')}")
        
        # List all document IDs
        if all_documents:
            doc_ids = [doc.get('document_id', 'NO_ID') for doc in all_documents]
            print(f"   Document IDs: {doc_ids}")
        
        # 2. Check TaskStateManager
        print("\n2. TaskStateManager Diagnostics:")
        try:
            from services.celery.task_state_manager import TaskStateManager
            
            task_manager = TaskStateManager()
            print(f"   TaskStateManager initialized: True")
            print(f"   Connection string: {task_manager.tasks_repo.connection_string}")
            
            # Test database connection
            try:
                from db.document_store.document_tasks_repository import DocumentTasksRepository
                test_repo = DocumentTasksRepository()
                
                # Try a simple query
                test_connection = test_repo._get_connection()
                with test_connection.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM document_tasks")
                    count = cursor.fetchone()[0]
                    print(f"   Database connection: OK")
                    print(f"   Total tasks in database: {count}")
                test_connection.close()
                
            except Exception as db_error:
                print(f"   Database connection: FAILED - {str(db_error)}")
            
            if document_id:
                task = task_manager.get_task_status(document_id)
                print(f"   Task for '{document_id}' found: {task is not None}")
                if task:
                    print(f"   Task status: {task.task_status}")
                    print(f"   Task stage: {task.current_stage}")
                    
        except Exception as e:
            print(f"   TaskStateManager error: {str(e)}")
        
        # 3. Check storage directories
        print("\n3. Storage Directory Diagnostics:")
        from core.config import get_config
        config = get_config()
        
        storage_dir = config.storage.storage_dir
        print(f"   Storage directory: {storage_dir}")
        print(f"   Storage directory exists: {os.path.exists(storage_dir)}")
        
        if os.path.exists(storage_dir):
            print(f"   Storage directory readable: {os.access(storage_dir, os.R_OK)}")
            print(f"   Storage directory writable: {os.access(storage_dir, os.W_OK)}")
            print(f"   Storage directory executable: {os.access(storage_dir, os.X_OK)}")
            
            # List subdirectories (should be document IDs)
            try:
                subdirs = [d for d in os.listdir(storage_dir) if os.path.isdir(os.path.join(storage_dir, d))]
                print(f"   Document subdirectories: {subdirs}")
                
                if document_id and document_id in subdirs:
                    doc_dir = os.path.join(storage_dir, document_id)
                    print(f"   Document '{document_id}' directory exists: True")
                    
                    # Check for expected files
                    original_pdf = os.path.join(doc_dir, "original.pdf")
                    print(f"   Original PDF exists: {os.path.exists(original_pdf)}")
                    
                    stages_dir = os.path.join(doc_dir, "stages")
                    print(f"   Stages directory exists: {os.path.exists(stages_dir)}")
                    
            except Exception as e:
                print(f"   Error listing storage directory: {str(e)}")
        
        # 4. Test document creation
        print("\n4. Document Creation Test:")
        test_doc_id = f"diagnostic_test_{int(datetime.now().timestamp())}"
        
        test_metadata = {
            "document_id": test_doc_id,
            "case_id": "diagnostic_test",
            "original_filename": "test.pdf",
            "status": "test",
            "test_created_at": datetime.now().isoformat()
        }
        
        print(f"   Testing document creation with ID: {test_doc_id}")
        
        # Try to add document
        add_success = doc_repo.add_document(test_metadata)
        print(f"   Document add success: {add_success}")
        
        if add_success:
            # Try to retrieve it
            retrieved_doc = doc_repo.get_document(test_doc_id)
            print(f"   Document retrieval success: {retrieved_doc is not None}")
            
            if retrieved_doc:
                print(f"   Retrieved document status: {retrieved_doc.get('status')}")
                
                # Try to update it
                update_success = doc_repo.update_document(test_doc_id, {"status": "updated"})
                print(f"   Document update success: {update_success}")
                
                # Clean up test document
                doc_repo.delete_document(test_doc_id)
                print(f"   Test document cleaned up")
            
        # 5. Check file permissions
        print("\n5. File Permissions Check:")
        current_dir = os.getcwd()
        print(f"   Current directory: {current_dir}")
        print(f"   Current directory writable: {os.access(current_dir, os.W_OK)}")
        
        # Check umask
        import stat
        test_file = "permission_test.tmp"
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            
            file_stat = os.stat(test_file)
            file_perms = stat.filemode(file_stat.st_mode)
            print(f"   Created file permissions: {file_perms}")
            
            os.remove(test_file)
        except Exception as e:
            print(f"   Permission test failed: {str(e)}")
        
        print("\n=== Diagnostics Complete ===")
        
    except Exception as e:
        print(f"Error in diagnostics: {str(e)}")
        import traceback
        traceback.print_exc()

def test_specific_document(document_id: str):
    """Test operations on a specific document"""
    
    print(f"\n=== Testing Document: {document_id} ===")
    
    try:
        from db.document_store.repository import DocumentMetadataRepository
        from services.celery.task_state_manager import TaskStateManager
        
        doc_repo = DocumentMetadataRepository()
        task_manager = TaskStateManager()
        
        # 1. Check document existence
        doc = doc_repo.get_document(document_id)
        print(f"Document in repository: {doc is not None}")
        
        if doc:
            print(f"  Status: {doc.get('status')}")
            print(f"  Case ID: {doc.get('case_id')}")
            print(f"  File path: {doc.get('stored_file_path')}")
            
            # Check if file exists
            file_path = doc.get('stored_file_path')
            if file_path:
                print(f"  File exists: {os.path.exists(file_path)}")
        
        # 2. Check task status
        task = task_manager.get_task_status(document_id)
        print(f"Task in task manager: {task is not None}")
        
        if task:
            print(f"  Task status: {task.task_status}")
            print(f"  Current stage: {task.current_stage}")
            print(f"  Can pause: {task.can_pause}")
            print(f"  Can resume: {task.can_resume}")
            print(f"  Can cancel: {task.can_cancel}")
        
    except Exception as e:
        print(f"Error testing document: {str(e)}")

if __name__ == "__main__":
    import sys
    
    document_id = None
    if len(sys.argv) > 1:
        document_id = sys.argv[1]
        print(f"Focusing on document: {document_id}")
    
    diagnose_document_registration(document_id)
    
    if document_id:
        test_specific_document(document_id)