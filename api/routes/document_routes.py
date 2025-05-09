# api/routes/document_routes.py
from fastapi import APIRouter, Depends, Query, HTTPException, Path
from fastapi.responses import FileResponse
from typing import List, Dict, Any
from db.document_store.repository import DocumentMetadataRepository
from api.routes.chat_routes import get_current_user, get_current_case
import os
from api.routes.admin_routes import get_admin_user
router = APIRouter(prefix="/ai/documents", tags=["documents"])

@router.get("/")
async def list_processed_documents(
    limit: int = Query(10, description="Number of documents per page"),
    offset: int = Query(0, description="Pagination offset"),
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case)
):
    """List all processed documents accessible to the current user"""
    doc_repository = DocumentMetadataRepository()
    
    # Get all documents
    all_documents = doc_repository.list_documents()
    
    # Apply pagination
    total = len(all_documents)
    paginated_documents = all_documents[offset:offset+limit]
    
    # Format response
    return {
        "documents": [
            {
                "document_id": doc.get("document_id", ""),
                "document_name": doc.get("original_filename", "Unnamed"),
                "status": doc.get("status", "Unknown"),
                "chunks_count": doc.get("chunks_count", 0),
                "processing_date": doc.get("processing_date")
            }
            for doc in paginated_documents
        ],
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset
        }
    }
    
@router.get("/{document_id}/file")
async def get_document_file(
    document_id: str = Path(..., description="Document ID"),
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case)
):
    """Get the original document file."""
    doc_repository = DocumentMetadataRepository()
    document = doc_repository.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get file path from metadata
    file_path = document.get("stored_file_path")
    if not file_path:
        file_path = document.get("original_file_path")
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Document file not found at path: {file_path}")
    
    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=document.get("original_filename", f"document-{document_id}.pdf")
    )
    
  
@router.get("/{document_id}")
async def get_document_details(
    document_id: str = Path(..., description="Document ID"),
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case)
):
    """Get detailed document information."""
    doc_repository = DocumentMetadataRepository()
    document = doc_repository.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document

@router.get("/documents/{document_id}", response_model=None)
async def get_admin_document_details(
    document_id: str,
    admin_user: str = Depends(get_admin_user)
):
    """Get detailed information about a specific document for admins."""
    doc_repository = DocumentMetadataRepository()
    document = doc_repository.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail=f"Document not found with ID: {document_id}")
    
    return document
