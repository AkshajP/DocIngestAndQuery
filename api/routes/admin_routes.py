# api/routes/admin_routes.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, Query
from typing import List, Dict, Any, Optional
import os
import time
import json
from datetime import datetime
from fastapi.responses import FileResponse
from api.models.admin_models import SystemStatsResponse, DocumentStats, SystemHealthStats, SystemStorageStats, QueryStats
from db.document_store.repository import DocumentMetadataRepository
from services.document.persistent_upload_service import PersistentUploadService
from services.document.processing_state_manager import ProcessingStage
from services.document.upload import upload_document
from core.config import get_config
from fastapi import HTTPException, Path
from api.routes.access_control import get_admin_user
import logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


router = APIRouter(prefix="/ai/admin", tags=["admin"])

@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(admin_user: str = Depends(get_admin_user)):
    """Get system-wide statistics including document processing stats."""
    doc_repository = DocumentMetadataRepository()
    stats = doc_repository.get_statistics()
    
    # Count documents by status
    all_documents = doc_repository.list_documents()
    processing_docs_count = len([
        doc for doc in all_documents 
        if doc.get("status") == "processing"
    ])
    pending_docs_count = len([
        doc for doc in all_documents 
        if doc.get("status") == "pending"
    ])
    
    # Total documents in queue = processing + pending
    queue_length = processing_docs_count + pending_docs_count
    
    # Calculate average processing time for completed documents
    processed_docs = [
        doc for doc in all_documents 
        if doc.get("status") == "processed" and doc.get("processing_time")
    ]
    
    avg_processing_time = 0.0
    if processed_docs:
        avg_processing_time = sum(doc.get("processing_time", 0) for doc in processed_docs) / len(processed_docs)
    
    # Count documents processed today
    today = datetime.now().date()
    processed_today = len([
        doc for doc in processed_docs 
        if doc.get("processing_date") and datetime.fromisoformat(doc.get("processing_date")).date() == today
    ])
    
    # Prepare document stats
    document_stats = DocumentStats(
        total_documents=stats["total_documents"],
        total_pages=stats["total_pages"],
        processed_today=processed_today,
        total_chunks=stats["total_chunks"],
        avg_processing_time=avg_processing_time
    )
    
    # Prepare query stats (placeholder values)
    query_stats = QueryStats(
        queries_today=0,
        total_queries=0,
        average_query_time=0.0,
        average_tokens_used=0
    )
    
    # Prepare system health stats
    system_health = SystemHealthStats(
        storage_usage=SystemStorageStats(
            total_gb=100.0,
            used_gb=10.0,
            available_gb=90.0,
            usage_percent=10.0
        ),
        vector_db_status="healthy",
        processing_queue_length=queue_length,
        active_jobs=0
    )
    
    return SystemStatsResponse(
        document_stats=document_stats,
        query_stats=query_stats,
        system_health=system_health
    )

@router.get("/documents")
async def list_all_documents(
    status: Optional[str] = None,
    admin_user: str = Depends(get_admin_user)
):
    """List all documents with detailed status information."""
    doc_repository = DocumentMetadataRepository()
    all_documents = doc_repository.list_documents()
    
    # Calculate status counts from ALL documents BEFORE filtering
    status_counts = {
        "processed": len([doc for doc in all_documents if doc.get("status") == "processed"]),
        "processing": len([doc for doc in all_documents if doc.get("status") == "processing"]), 
        "failed": len([doc for doc in all_documents if doc.get("status") == "failed"]),
        "pending": len([doc for doc in all_documents if doc.get("status") == "pending"])
    }
    
    # NOW filter documents for display (keep original all_documents intact)
    filtered_documents = all_documents
    if status and status != "all":
        filtered_documents = [doc for doc in all_documents if doc.get("status") == status]
    
    formatted_documents = []
    for doc in filtered_documents:
        formatted_doc = {
            "document_id": doc.get("document_id", ""),
            "document_name": doc.get("original_filename", "Unnamed"),  # ✅ Map original_filename to document_name
            "status": doc.get("status", "Unknown"),
            "chunks_count": doc.get("chunks_count", 0),
            "processing_date": doc.get("processing_date"),
            "case_path": doc.get("case_path", None),
            "page_count": doc.get("page_count", None),
            "total_processing_time": doc.get("total_processing_time"),
            "processing_state": doc.get("processing_state"),
            # Include original fields for admin functionality
            "original_filename": doc.get("original_filename"),
            "original_file_path": doc.get("original_file_path"),
            "stored_file_path": doc.get("stored_file_path"),
            "case_id": doc.get("case_id"),
            "file_type": doc.get("file_type"),
            "user_metadata": doc.get("user_metadata", {}),
        }
        formatted_documents.append(formatted_doc)
    
    return {
        "documents": formatted_documents,  # ✅ Now properly formatted
        "total": len(formatted_documents),
        "status_counts": status_counts  # ✅ Always based on all documents, not filtered ones
    }
    
    
@router.post("/documents/upload")
async def upload_new_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    case_id: str = Form("default"),
    admin_user: str = Depends(get_admin_user)
):
    """Upload and process a new document using Celery tasks."""
    temp_file_path = f"/tmp/{file.filename}"
    
    try:
        # Save uploaded file to temporary location
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        # Create document ID
        timestamp = int(time.time())
        safe_name = ''.join(c for c in file.filename.split('.')[0].replace(' ', '_') 
                          if c.isalnum() or c == '_')
        document_id = f"doc_{timestamp}_{safe_name}"
        
        # Start Celery processing chain
        from services.document.upload import upload_document_with_celery
        result = upload_document_with_celery(
            file_path=temp_file_path,
            document_id=document_id,
            case_id=case_id
        )
        
        if result["status"] == "processing":
            return {
                "status": "processing",
                "message": "Document upload initiated. Processing started with Celery.",
                "document_id": document_id,
                "celery_task_id": result["celery_task_id"],
                "task_control": result["task_control"]
            }
        else:
            return result
        
    except Exception as e:
        # Clean up temp file
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@router.post("/documents/{document_id}/retry")
async def retry_document_processing(
    document_id: str,
    background_tasks: BackgroundTasks,
    from_stage: Optional[str] = Query(None, description="Stage to retry from (optional)"),
    force_restart: bool = Query(False, description="Force restart from beginning"),
    admin_user: str = Depends(get_admin_user)
):
    """Retry processing a document from a specific stage or current failed stage."""
    doc_repository = DocumentMetadataRepository()
    document = doc_repository.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check for concurrent processing
    if document.get("status") == "processing":
        # Check if there's an active processing lock
        upload_service = PersistentUploadService()
        doc_dir = os.path.join(upload_service.config.storage.storage_dir, document_id)
        lock_file = os.path.join(doc_dir, "stages", "processing.lock")
        
        if upload_service.storage_adapter.file_exists(lock_file):
            try:
                lock_content = upload_service.storage_adapter.read_file(lock_file)
                if lock_content:
                    lock_data = json.loads(lock_content)
                    locked_at = datetime.fromisoformat(lock_data["locked_at"])
                    if (datetime.now() - locked_at).total_seconds() < 300:  # 5 minutes
                        raise HTTPException(
                            status_code=409, 
                            detail="Document is currently being processed. Please wait before retrying."
                        )
            except:
                pass  # If we can't read lock, proceed
    
    # Validate stage if provided
    if from_stage:
        valid_stages = [stage.value for stage in ProcessingStage]
        if from_stage not in valid_stages:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid stage. Valid stages: {', '.join(valid_stages)}"
            )
    
    # Get original file path
    file_path = document.get("stored_file_path") or document.get("original_file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Original file not found")
    
    # Update document status to processing with retry information
    retry_metadata = {
        "status": "processing",
        "retry_time": datetime.now().isoformat(),
        "retry_count": document.get("retry_count", 0) + 1,
        "retry_from_stage": from_stage,
        "force_restart": force_restart
    }
    
    doc_repository.update_document(document_id, retry_metadata)
    
    # Queue document processing in background
    if force_restart:
        # Use the persistent upload service with force restart
        background_tasks.add_task(
            _retry_with_persistent_service,
            file_path=file_path,
            document_id=document_id,
            case_id=document.get("case_id", "default"),
            force_restart=True
        )
    else:
        # Use retry functionality
        background_tasks.add_task(
            _retry_from_stage,
            document_id=document_id,
            from_stage=from_stage,
            case_id=document.get("case_id", "default")
        )
    
    return {
        "status": "processing",
        "message": f"Document processing {'restarted' if force_restart else 'retried'}",
        "document_id": document_id,
        "from_stage": from_stage if not force_restart else "upload",
        "retry_count": retry_metadata["retry_count"]
    }
    
@router.get("/documents/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    chunk_type: str = Query("original", description="Type of chunks to retrieve"),
    page_number: Optional[int] = Query(None, description="Filter by page number (1-based)"),
    admin_user: str = Depends(get_admin_user)
):
    """Get all chunks for a specific document with their bounding boxes."""
    # Get document details to check case_id
    doc_repository = DocumentMetadataRepository()
    document = doc_repository.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    case_id = document.get("case_id", "default")
    
    # Convert 1-based page_number from frontend to 0-based for vector DB
    db_page_number = None
    if page_number is not None:
        db_page_number = page_number - 1  # Convert from 1-based to 0-based
    
    try:
        # Initialize vector store
        from db.vector_store.adapter import VectorStoreAdapter
        
        # We need a dummy embedding for search
        dummy_embedding = [0.0] * 3072
        
        # Initialize the vector store adapter
        vector_store = VectorStoreAdapter()
        
        # Use the search method to get chunks
        chunks = vector_store.search(
            query_embedding=dummy_embedding,
            case_ids=[case_id],
            document_ids=[document_id],
            chunk_types=[chunk_type],
            top_k=1000  # Set a high value to get all chunks
        )
        
        # Format response with chunk content and bounding boxes
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Extract bounding boxes from metadata
            original_boxes = None
            if chunk.get("metadata") and "original_boxes" in chunk["metadata"]:
                original_boxes = chunk["metadata"].get("original_boxes")
            
            # Try to get bbox from metadata if original_boxes is not available
            if not original_boxes and chunk.get("metadata") and "bbox" in chunk["metadata"]:
                original_boxes = [chunk["metadata"]["bbox"]]
            
            # Get the page number from the chunk (0-based in DB)
            chunk_db_page = chunk.get("page_number")
            
            # Filter by db_page_number if specified
            if db_page_number is not None and chunk_db_page != db_page_number:
                continue
            
            # Convert 0-based DB page to 1-based frontend page
            frontend_page_number = chunk_db_page + 1 if chunk_db_page is not None else None
                
            # Ensure each chunk has a unique ID
            chunk_id = chunk.get("chunk_id", f"chunk_{document_id}_{i}")
                
            formatted_chunks.append({
                "chunk_id": chunk_id,
                "content": chunk.get("content", ""),
                "page_number": frontend_page_number-1,  # 1-based page number for frontend
                "bounding_boxes": original_boxes,
                "metadata": chunk.get("metadata", {})
            })
        
        # Sort by page number and position on page
        formatted_chunks.sort(key=lambda x: (
            x.get("page_number", 0),
            x.get("metadata", {}).get("bbox", [0, 0, 0, 0])[1] if x.get("metadata", {}).get("bbox") else 0
        ))
        
        return {
            "document_id": document_id,
            "chunk_type": chunk_type,
            "chunks_count": len(formatted_chunks),
            "chunks": formatted_chunks
        }
    
    except Exception as e:
        logger.error(f"Error retrieving document chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document chunks: {str(e)}")

@router.get("/documents/{document_id}/processing-status")
async def get_document_processing_status(
    document_id: str,
    admin_user: str = Depends(get_admin_user)
):
    """Get detailed processing status for a specific document."""
    upload_service = PersistentUploadService()
    result = upload_service.get_document_processing_status(document_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result

@router.get("/documents/{document_id}/stages")
async def get_document_stages(
    document_id: str,
    admin_user: str = Depends(get_admin_user)
):
    """Get status of all processing stages for a document."""
    upload_service = PersistentUploadService()
    result = upload_service.get_document_processing_status(document_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["error"])
    
    processing_status = result["processing_status"]
    
    # Format stage information for easier consumption
    all_stages = [stage.value for stage in ProcessingStage]
    stage_info = []
    
    for stage in all_stages:
        is_completed = stage in processing_status.get("completed_stages", [])
        completion_time = processing_status.get("stage_completion_times", {}).get(stage)
        error_details = processing_status.get("stage_error_details", {}).get(stage)
        retry_count = processing_status.get("retry_counts", {}).get(stage, 0)
        
        stage_info.append({
            "stage": stage,
            "status": "completed" if is_completed else "pending",
            "is_current": stage == processing_status.get("current_stage"),
            "completion_time": completion_time,
            "error_details": error_details,
            "retry_count": retry_count
        })
    
    return {
        "document_id": document_id,
        "current_stage": processing_status.get("current_stage"),
        "last_updated": processing_status.get("last_updated"),
        "stages": stage_info
    }

@router.post("/documents/{document_id}/stages/{stage}/reset")
async def reset_document_to_stage(
    document_id: str,
    stage: str,
    background_tasks: BackgroundTasks,
    admin_user: str = Depends(get_admin_user)
):
    """Reset document processing to a specific stage, clearing subsequent stages."""
    # Validate stage
    valid_stages = [stage_enum.value for stage_enum in ProcessingStage]
    if stage not in valid_stages:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid stage. Valid stages: {', '.join(valid_stages)}"
        )
    
    doc_repository = DocumentMetadataRepository()
    document = doc_repository.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check for concurrent processing
    if document.get("status") == "processing":
        upload_service = PersistentUploadService()
        doc_dir = os.path.join(upload_service.config.storage.storage_dir, document_id)
        lock_file = os.path.join(doc_dir, "stages", "processing.lock")
        
        if upload_service.storage_adapter.file_exists(lock_file):
            try:
                lock_content = upload_service.storage_adapter.read_file(lock_file)
                if lock_content:
                    lock_data = json.loads(lock_content)
                    locked_at = datetime.fromisoformat(lock_data["locked_at"])
                    if (datetime.now() - locked_at).total_seconds() < 300:  # 5 minutes
                        raise HTTPException(
                            status_code=409, 
                            detail="Document is currently being processed. Cannot reset stage."
                        )
            except:
                pass  # If we can't read lock, proceed
    
    # Initialize upload service and reset
    upload_service = PersistentUploadService()
    
    try:
        # Get document paths for state manager initialization
        doc_dir = os.path.join(upload_service.config.storage.storage_dir, document_id)
        
        # Initialize processing state manager
        from services.document.processing_state_manager import ProcessingStateManager
        state_manager = ProcessingStateManager(
            document_id=document_id,
            storage_adapter=upload_service.storage_adapter,
            doc_dir=doc_dir,
            doc_repository=doc_repository
        )
        
        # Reset to specified stage
        success = state_manager.reset_to_stage(stage)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to reset document stage")
        
        # Update document metadata
        doc_repository.update_document(document_id, {
            "status": "processing",
            "current_stage": stage,
            "reset_time": datetime.now().isoformat(),
            "processing_state": state_manager.get_stage_status()
        })
        
        return {
            "status": "success",
            "message": f"Document reset to stage '{stage}'",
            "document_id": document_id,
            "new_current_stage": stage
        }
        
    except Exception as e:
        logger.error(f"Error resetting document {document_id} to stage {stage}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting document: {str(e)}")

@router.delete("/documents/{document_id}/stages/{stage}/data")
async def cleanup_stage_data(
    document_id: str,
    stage: str,
    admin_user: str = Depends(get_admin_user)
):
    """Clean up intermediate data for a specific stage."""
    # Validate stage
    valid_stages = [stage_enum.value for stage_enum in ProcessingStage]
    if stage not in valid_stages:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid stage. Valid stages: {', '.join(valid_stages)}"
        )
    
    doc_repository = DocumentMetadataRepository()
    document = doc_repository.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Initialize upload service
        upload_service = PersistentUploadService()
        doc_dir = os.path.join(upload_service.config.storage.storage_dir, document_id)
        
        # Initialize processing state manager
        from services.document.processing_state_manager import ProcessingStateManager
        state_manager = ProcessingStateManager(
            document_id=document_id,
            storage_adapter=upload_service.storage_adapter,
            doc_dir=doc_dir,
            doc_repository=doc_repository
        )
        
        # Clean up stage data
        success = state_manager.cleanup_stage_data(stage)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to cleanup stage data")
        
        return {
            "status": "success",
            "message": f"Cleaned up data for stage '{stage}'",
            "document_id": document_id,
            "stage": stage
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up stage {stage} for document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cleaning up stage data: {str(e)}")
    
@router.get("/documents/processing-stats")
async def get_processing_statistics(
    admin_user: str = Depends(get_admin_user)
):
    """Get statistics about document processing stages across all documents."""
    doc_repository = DocumentMetadataRepository()
    
    try:
        # Use the enhanced statistics method from repository
        detailed_stats = doc_repository.get_processing_statistics_detailed()
        
        return {
            "total_documents": detailed_stats["total_documents"],
            "by_status": detailed_stats["by_status"],
            "by_current_stage": detailed_stats["by_current_stage"],
            "stage_error_counts": detailed_stats["stage_error_counts"],
            "retry_statistics": detailed_stats["retry_statistics"],
            "last_updated": detailed_stats["last_updated"]
        }
    except Exception as e:
        logger.error(f"Error getting processing statistics: {str(e)}")
        # Fallback to basic statistics
        all_documents = doc_repository.list_documents()
        
        stats = {
            "total_documents": len(all_documents),
            "by_status": {},
            "by_current_stage": {},
            "stage_error_counts": {},
            "retry_statistics": {},
            "processing_times": {
                "avg_total": 0,
                "avg_by_stage": {}
            }
        }
        
        # Collect basic statistics
        total_processing_times = []
        stage_processing_times = {}
        
        for doc in all_documents:
            status = doc.get("status", "unknown")
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            
            # Get processing state if available
            processing_state = doc.get("processing_state", {})
            current_stage = processing_state.get("current_stage", "unknown")
            stats["by_current_stage"][current_stage] = stats["by_current_stage"].get(current_stage, 0) + 1
            
            # Failed stages
            stage_errors = processing_state.get("stage_error_details", {})
            for stage, error in stage_errors.items():
                if stage not in stats["stage_error_counts"]:
                    stats["stage_error_counts"][stage] = 0
                stats["stage_error_counts"][stage] += 1
            
            # Retry counts
            retry_counts = processing_state.get("retry_counts", {})
            for stage, count in retry_counts.items():
                if stage not in stats["retry_statistics"]:
                    stats["retry_statistics"][stage] = []
                stats["retry_statistics"][stage].append(count)
            
            # Processing times
            total_time = doc.get("total_processing_time")
            if total_time:
                total_processing_times.append(total_time)
        
        # Calculate averages
        if total_processing_times:
            stats["processing_times"]["avg_total"] = sum(total_processing_times) / len(total_processing_times)
        
        # Average retry counts
        for stage, counts in stats["retry_statistics"].items():
            stats["retry_statistics"][stage] = {
                "avg_retries_per_doc": sum(counts) / len(counts),
                "total_retries": sum(counts),
                "max_retries": max(counts),
                "documents_with_retries": len(counts)
            }
        
        return stats

# Background task functions
async def _retry_with_persistent_service(file_path: str, document_id: str, case_id: str, force_restart: bool = False):
    """Background task to retry document processing with persistent service"""
    try:
        upload_service = PersistentUploadService()
        result = upload_service.upload_document(
            file_path=file_path,
            document_id=document_id,
            case_id=case_id,
            force_restart=force_restart
        )
        logger.info(f"Retry processing completed for document {document_id}: {result['status']}")
    except Exception as e:
        logger.error(f"Error in retry processing for document {document_id}: {str(e)}")

async def _retry_from_stage(document_id: str, from_stage: Optional[str], case_id: str):
    """Background task to retry document processing from specific stage"""
    try:
        upload_service = PersistentUploadService()
        result = upload_service.retry_document_processing(
            document_id=document_id,
            from_stage=from_stage,
            case_id=case_id
        )
        logger.info(f"Retry from stage completed for document {document_id}: {result['status']}")
    except Exception as e:
        logger.error(f"Error in retry from stage for document {document_id}: {str(e)}")

@router.post("/documents/{document_id}/force-unlock")
async def force_unlock_document(
    document_id: str,
    admin_user: str = Depends(get_admin_user)
):
    """Force remove processing locks for a stuck document after server crash"""
    try:
        upload_service = PersistentUploadService()
        doc_dir = os.path.join(upload_service.config.storage.storage_dir, document_id)
        
        # Remove processing lock
        lock_file = os.path.join(doc_dir, "stages", "processing.lock")
        state_lock_file = os.path.join(doc_dir, "stages", "state.lock")
        
        locks_removed = []
        
        if upload_service.storage_adapter.file_exists(lock_file):
            upload_service.storage_adapter.delete_file(lock_file)
            locks_removed.append("processing.lock")
        
        if upload_service.storage_adapter.file_exists(state_lock_file):
            upload_service.storage_adapter.delete_file(state_lock_file)
            locks_removed.append("state.lock")
        
        # Reset document status if it was stuck in processing
        doc_repository = DocumentMetadataRepository()
        document = doc_repository.get_document(document_id)
        
        if document and document.get("status") == "processing":
            # Reset to a recoverable state
            processing_state = document.get("processing_state", {})
            current_stage = processing_state.get("current_stage", "upload")
            
            doc_repository.update_document(document_id, {
                "status": "failed",  # Mark as failed so it can be retried
                "error_message": "Server crash recovery - locks removed",
                "recovery_time": datetime.now().isoformat()
            })
        
        return {
            "status": "success", 
            "message": f"Removed {len(locks_removed)} lock files",
            "locks_removed": locks_removed,
            "document_status_reset": document.get("status") == "processing" if document else False
        }
        
    except Exception as e:
        logger.error(f"Error force unlocking document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unlock document: {str(e)}")

@router.post("/system/cleanup-stale-locks")
async def cleanup_all_stale_locks(
    admin_user: str = Depends(get_admin_user)
):
    """Clean up all stale locks across the system"""
    try:
        upload_service = PersistentUploadService()
        cleanup_result = upload_service.cleanup_stale_locks()
        
        return {
            "status": "success",
            "message": "Stale locks cleanup completed",
            **cleanup_result
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up stale locks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup stale locks: {str(e)}")