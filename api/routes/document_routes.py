# api/routes/document_routes.py
from fastapi import APIRouter, Depends, Query, HTTPException, Path, Response
from fastapi.responses import FileResponse
from typing import List, Dict, Any
from db.document_store.repository import DocumentMetadataRepository
from api.routes.access_control import get_current_user, get_current_case, validate_user_case_access
import os
import fitz 
import logging
logger = logging.getLogger(__name__)
from api.routes.admin_routes import get_admin_user
router = APIRouter(prefix="/ai/documents", tags=["documents"])

@router.get("/")
async def list_processed_documents(
    limit: int = Query(100, description="Number of documents per page"),
    offset: int = Query(0, description="Pagination offset"),
    case_id: str = Depends(get_current_case),
    _: bool = Depends(validate_user_case_access)
):
    """List all processed documents accessible to the current user"""
    doc_repository = DocumentMetadataRepository()
    
    # Get all documents
    all_documents = doc_repository.list_documents_by_case(case_id=case_id)
    
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
                "processing_date": doc.get("processing_date"),
                "case_path": doc.get("case_path", None),
                "page_count": doc.get("page_count", None)
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
    _: bool = Depends(validate_user_case_access)
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
    _: bool = Depends(validate_user_case_access)
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

@router.get("/{document_id}/page/{page_number}")
async def get_document_page(
    document_id: str = Path(..., description="Document ID"),
    page_number: int = Path(..., description="Page number (1-based)"),
    _: bool = Depends(validate_user_case_access)
):
    """
    Get a specific page from the document as a PDF.
    This extracts only the requested page and returns it as a linearized PDF.
    """
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
    
    try:
        # Convert page_number from 1-based to 0-based for PyMuPDF
        page_index = page_number - 1
        
        # Open the PDF
        pdf_document = fitz.open(file_path)
        
        # Validate page number
        if page_index < 0 or page_index >= len(pdf_document):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid page number. Document has {len(pdf_document)} pages."
            )
        
        # Create a new PDF with just the requested page
        output_pdf = fitz.open()
        output_pdf.insert_pdf(pdf_document, from_page=page_index, to_page=page_index)
        
        # Get the PDF as bytes
        pdf_bytes = output_pdf.tobytes(
            deflate=True,   # Compress the output
            garbage=3,      # Garbage collection level
            clean=True,     # Remove unnecessary elements
            linear=True     # Linearize the PDF for streaming
        )
        
        # Clean up
        output_pdf.close()
        pdf_document.close()
        
        # Return the PDF page
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="document-{document_id}-page-{page_number}.pdf"'
            }
        )
    except Exception as e:
        logger.error(f"Error extracting PDF page: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting PDF page: {str(e)}")
