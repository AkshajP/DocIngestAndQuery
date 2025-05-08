# api/routes/document_routes.py
from fastapi import APIRouter, Depends, Query
from typing import List, Dict, Any
from db.document_store.repository import DocumentMetadataRepository
from api.routes.chat_routes import get_current_user, get_current_case

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