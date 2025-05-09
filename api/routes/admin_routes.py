# api/routes/admin_routes.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import List, Dict, Any, Optional
import os
import time
from datetime import datetime
from fastapi.responses import FileResponse
from api.models.admin_models import SystemStatsResponse, DocumentStats, SystemHealthStats, SystemStorageStats, QueryStats
from db.document_store.repository import DocumentMetadataRepository
from services.document.upload import upload_document
from core.config import get_config
from fastapi import HTTPException, Path
from api.routes.chat_routes import get_current_user, get_current_case
import logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
# Dependency for admin authentication
async def get_admin_user():
    # This would normally verify the token and ensure admin privileges
    return "admin_user"

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
    
    # Filter by status if requested
    if status and status != "all":
        all_documents = [doc for doc in all_documents if doc.get("status") == status]
    
    # Format the response
    return {
        "documents": all_documents,
        "total": len(all_documents),
        "status_counts": {
            "processed": len([doc for doc in all_documents if doc.get("status") == "processed"]),
            "processing": len([doc for doc in all_documents if doc.get("status") == "processing"]),
            "failed": len([doc for doc in all_documents if doc.get("status") == "failed"]),
            "pending": len([doc for doc in all_documents if doc.get("status") == "pending"])
        }
    }

@router.post("/documents/upload")
async def upload_new_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    case_id: str = Form("default"),
    admin_user: str = Depends(get_admin_user)
):
    """Upload and process a new document."""
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
        
        # Queue document processing in background
        background_tasks.add_task(
            upload_document,
            file_path=temp_file_path,
            document_id=document_id,
            case_id=case_id
        )
        
        return {
            "status": "pending",
            "message": "Document upload initiated. Processing will start shortly.",
            "document_id": document_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@router.post("/documents/{document_id}/retry")
async def retry_document_processing(
    document_id: str,
    background_tasks: BackgroundTasks,
    admin_user: str = Depends(get_admin_user)
):
    """Retry processing a failed document."""
    doc_repository = DocumentMetadataRepository()
    document = doc_repository.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document.get("status") not in ["failed", "processing"]:
        raise HTTPException(status_code=400, detail="Document is not in a failed or stuck processing state")
    
    # Get original file path
    file_path = document.get("original_file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Original file not found")
    
    # Update document status to pending
    doc_repository.update_document(document_id, {
        "status": "pending",
        "retry_count": document.get("retry_count", 0) + 1,
        "retry_time": datetime.now().isoformat()
    })
    
    # Queue document processing in background
    background_tasks.add_task(
        upload_document,
        file_path=file_path,
        document_id=document_id,
        case_id=document.get("case_id", "default")
    )
    
    return {
        "status": "pending",
        "message": "Document processing restarted",
        "document_id": document_id
    }
    

  