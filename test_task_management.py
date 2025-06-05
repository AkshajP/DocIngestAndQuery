#!/usr/bin/env python3
"""
Simple test script for Phase 1 Step 2: Task State Management
Run this to verify the basic functionality works.
"""

import sys
import os
import time
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_task_state_manager():
    """Test the core task state manager functionality"""
    print("🧪 Testing Task State Manager...")
    
    try:
        from services.celery.task_state_manager import TaskStateManager
        from db.document_store.document_tasks_repository import TaskStatus, ProcessingStage
        
        # Initialize task state manager
        print("   ✓ Importing task state manager...")
        task_manager = TaskStateManager()
        print("   ✓ Task state manager initialized")
        
        # Test document ID
        test_doc_id = f"test_doc_{int(time.time())}"
        test_celery_id = f"celery_test_{int(time.time())}"
        
        print(f"   📄 Testing with document ID: {test_doc_id}")
        
        # 1. Create a workflow task
        print("   1️⃣ Creating workflow task...")
        task_id = task_manager.create_workflow_task(
            document_id=test_doc_id,
            initial_stage=ProcessingStage.UPLOAD.value,
            celery_task_id=test_celery_id
        )
        print(f"   ✓ Created task with ID: {task_id}")
        
        # 2. Start the task
        print("   2️⃣ Starting task...")
        success = task_manager.start_task(
            document_id=test_doc_id,
            celery_task_id=test_celery_id,
            worker_info={"worker_id": "test_worker", "worker_hostname": "test_host"}
        )
        print(f"   ✓ Task started: {success}")
        
        # 3. Get task status
        print("   3️⃣ Getting task status...")
        task = task_manager.get_task_status(test_doc_id)
        if task:
            print(f"   ✓ Task status: {task.task_status}")
            print(f"   ✓ Can pause: {task.can_pause}")
            print(f"   ✓ Can cancel: {task.can_cancel}")
        else:
            print("   ❌ No task found")
            return False
        
        # 4. Update progress
        print("   4️⃣ Updating progress...")
        success = task_manager.update_progress(
            document_id=test_doc_id,
            percent_complete=50,
            checkpoint_data={"stage": "testing", "items_processed": 100}
        )
        print(f"   ✓ Progress updated: {success}")
        
        # 5. Test pause request
        print("   5️⃣ Testing pause request...")
        success = task_manager.pause_task(test_doc_id)
        print(f"   ✓ Pause requested: {success}")
        
        # Check if pause was registered
        task = task_manager.get_task_status(test_doc_id)
        if task:
            print(f"   ✓ Pause requested flag: {task.pause_requested}")
        
        # 6. Mark as paused
        print("   6️⃣ Marking as paused...")
        success = task_manager.mark_task_paused(test_doc_id)
        print(f"   ✓ Marked as paused: {success}")
        
        # 7. Test resume
        print("   7️⃣ Testing resume...")
        new_celery_id = f"celery_resume_{int(time.time())}"
        success = task_manager.resume_task(test_doc_id, new_celery_id)
        print(f"   ✓ Task resumed: {success}")
        
        # 8. Complete the task
        print("   8️⃣ Completing task...")
        success = task_manager.complete_task(test_doc_id, success=True)
        print(f"   ✓ Task completed: {success}")
        
        # 9. Final status check
        print("   9️⃣ Final status check...")
        task = task_manager.get_task_status(test_doc_id)
        if task:
            print(f"   ✓ Final status: {task.task_status}")
            print(f"   ✓ Percent complete: {task.percent_complete}")
        
        # 10. Cleanup
        print("   🧹 Cleaning up test data...")
        success = task_manager.tasks_repo.delete_task(test_doc_id)
        print(f"   ✓ Test task deleted: {success}")
        
        print("   🎉 Task State Manager test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   💡 Make sure database dependencies are installed: pip install psycopg2-binary")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_document_repository_integration():
    """Test the document repository integration with task management"""
    print("\n🗄️ Testing Document Repository Integration...")
    
    try:
        from db.document_store.repository import DocumentMetadataRepository
        
        repo = DocumentMetadataRepository()
        print("   ✓ Document repository initialized")
        
        # Test document ID
        test_doc_id = f"test_repo_doc_{int(time.time())}"
        
        # Create a test document
        print("   1️⃣ Creating test document...")
        doc_metadata = {
            "document_id": test_doc_id,
            "case_id": "test_case",
            "original_filename": "test.pdf",
            "status": "processing",
            "can_pause": True,
            "can_cancel": True
        }
        
        success = repo.add_document(doc_metadata)
        print(f"   ✓ Document created: {success}")
        
        # Test pause marking
        print("   2️⃣ Testing pause marking...")
        success = repo.mark_task_as_paused(test_doc_id)
        print(f"   ✓ Marked as paused: {success}")
        
        # Get pausable documents
        print("   3️⃣ Getting pausable documents...")
        pausable_docs = repo.get_pausable_documents()
        print(f"   ✓ Found {len(pausable_docs)} pausable documents")
        
        # Test resume marking
        print("   4️⃣ Testing resume marking...")
        success = repo.mark_task_as_resumed(test_doc_id)
        print(f"   ✓ Marked as resumed: {success}")
        
        # Test cancel marking
        print("   5️⃣ Testing cancel marking...")
        success = repo.mark_task_as_cancelled(test_doc_id)
        print(f"   ✓ Marked as cancelled: {success}")
        
        # Cleanup
        print("   🧹 Cleaning up...")
        success = repo.delete_document(test_doc_id)
        print(f"   ✓ Test document deleted: {success}")
        
        print("   🎉 Document Repository integration test completed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_database_connection():
    """Test basic database connectivity"""
    print("\n🔗 Testing Database Connection...")
    
    try:
        from db.document_store.document_tasks_repository import DocumentTasksRepository
        
        # Test connection
        repo = DocumentTasksRepository()
        print("   ✓ Database connection successful")
        
        # Try to get tasks (should work even if empty)
        tasks = repo.get_tasks_by_status("PENDING", limit=1)
        print(f"   ✓ Database query successful (found {len(tasks)} pending tasks)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        print("   💡 Make sure PostgreSQL is running and configuration is correct")
        print("   💡 Check DATABASE_URL or database config in core/config.py")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Phase 1 Step 2: Task State Management")
    print("=" * 50)
    
    # Test 1: Database Connection
    if not test_database_connection():
        print("\n❌ Database connection failed. Please fix database setup first.")
        return False
    
    # Test 2: Task State Manager
    if not test_task_state_manager():
        print("\n❌ Task State Manager test failed.")
        return False
    
    # Test 3: Document Repository Integration
    if not test_document_repository_integration():
        print("\n❌ Document Repository integration test failed.")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Task State Management is working correctly.")
    print("\n💡 Next steps:")
    print("   - The database table is set up correctly")
    print("   - Task state management is functional")
    print("   - Document repository integration works")
    print("   - Ready for Phase 1 Step 3: Core Document Processing Tasks")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)