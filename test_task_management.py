#!/usr/bin/env python3
"""
Example script showing how to use the new Celery-based document processing system.

This demonstrates:
1. Starting a document processing chain
2. Checking task status  
3. Pausing/resuming tasks
4. Cancelling tasks

Usage:
    python example_celery_usage.py

Requirements:
    - Redis running
    - Celery worker running: python worker.py  
    - Document file available for testing
"""
from datetime import datetime
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def example_document_processing():
    """Example of processing a document with Celery"""
    
    print("=== Celery Document Processing Example ===")
    
    # Test document path - adjust this to your test file
    test_file = '/Users/vikas/Downloads/Index Volume 18.pdf'
    
    if not os.path.exists(test_file):
        print(f"Please create a test file at {test_file} or update the path")
        return
    
    try:
        # Import after path setup
        from services.document.upload import upload_document_with_celery
        from services.celery.task_state_manager import TaskStateManager
        
        print("1. Starting document processing with Celery...")
        
        # Start processing
        result = upload_document_with_celery(
            file_path=test_file,
            document_id=f"test_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            case_id="test_case",
            metadata={"source": "example_script"}
        )
        
        print(f"Upload result: {result}")
        
        if result["status"] != "processing":
            print(f"Failed to start processing: {result}")
            return
        
        document_id = result["document_id"]
        celery_task_id = result["celery_task_id"]
        
        print(f"Document ID: {document_id}")
        print(f"Celery Task ID: {celery_task_id}")
        
        # Initialize task state manager for monitoring
        task_manager = TaskStateManager()
        
        print("\n2. Monitoring progress...")
        
        # Monitor for a bit
        for i in range(10):
            time.sleep(2)
            
            # Get task status
            task = task_manager.get_task_status(document_id)
            if task:
                print(f"  Progress: {task.percent_complete}% - Status: {task.task_status} - Stage: {task.current_stage}")
                
                # Demonstrate pause after 20% progress
                # if i == 3 and task.can_pause:
                    # print("\n3. Demonstrating pause...")
                    # pause_result = task_manager.pause_task(document_id)
                    # print(f"  Pause result: {pause_result}")
                    
                    # time.sleep(10)
                    
                    # print("\n4. Demonstrating resume...")
                    # resume_result = task_manager.resume_task(document_id, f"resume_{celery_task_id}")
                    # print(f"  Resume result: {resume_result}")
                
                # Check if completed
                if task.task_status in ["SUCCESS", "FAILURE", "CANCELLED"]:
                    print(f"\n5. Processing completed with status: {task.task_status}")
                    break
            else:
                print(f"  No task status found for {document_id}")
        
        print("\n=== Example completed ===")
        
    except ImportError as e:
        print(f"Import error - make sure all services are available: {e}")
    except Exception as e:
        print(f"Error: {e}")

def example_task_control():
    """Example of controlling running tasks"""
    
    print("\n=== Task Control Example ===")
    
    try:
        from services.celery.task_state_manager import TaskStateManager
        
        task_manager = TaskStateManager()
        
        # Get active tasks
        active_tasks = task_manager.get_active_tasks()
        print(f"Found {len(active_tasks)} active tasks")
        
        if active_tasks:
            task = active_tasks[0]
            document_id = task.document_id
            
            print(f"Controlling task for document: {document_id}")
            print(f"Current status: {task.task_status}")
            print(f"Can pause: {task.can_pause}")
            print(f"Can resume: {task.can_resume}")
            print(f"Can cancel: {task.can_cancel}")
            
            # Example: Cancel the first active task
            # Uncomment to actually cancel:
            # if task.can_cancel:
            #     print("Cancelling task...")
            #     result = task_manager.cancel_task(document_id)
            #     print(f"Cancel result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Starting Celery document processing examples...")
    print("Make sure you have:")
    print("1. Redis running")
    print("2. Celery worker running: python worker.py")
    print("3. Database tables created")
    print()
    
    # Run examples
    example_document_processing()
    example_task_control()
    
    print("\nDone!")