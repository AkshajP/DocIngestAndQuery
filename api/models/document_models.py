from pydantic import BaseModel, Field, validator, model_validator
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from datetime import datetime
import uuid


class DocumentStatus(str, Enum):
    """Status of a document in the processing pipeline"""
    PENDING = "pending"
    PROCESSING = "processing"
    FAILED = "failed"
    PROCESSED = "processed"
    DELETED = "deleted"


class ProcessingStageStatus(str, Enum):
    """Status of an individual processing stage"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProcessingStage(BaseModel):
    """Information about a specific processing stage"""
    status: ProcessingStageStatus
    time_taken: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    # These fields will vary depending on the stage
    pages_processed: Optional[int] = None
    chunks_created: Optional[int] = None
    embeddings_generated: Optional[int] = None
    tree_levels: Optional[int] = None
    progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")


class DocumentProcessingStages(BaseModel):
    """Collection of all processing stages for a document"""
    mineru: Optional[ProcessingStage] = None
    chunking: Optional[ProcessingStage] = None
    embeddings: Optional[ProcessingStage] = None
    raptor: Optional[ProcessingStage] = None


class DocumentMetadata(BaseModel):
    """Basic document metadata"""
    document_id: str = Field(..., description="Unique document identifier")
    original_filename: str = Field(..., description="Original filename")
    status: DocumentStatus
    processing_date: Optional[datetime] = None
    page_count: Optional[int] = None
    chunks_count: Optional[int] = None
    file_type: Optional[str] = None
    language: Optional[str] = None
    content_types: Optional[Dict[str, int]] = None
    raptor_levels: Optional[List[int]] = None
    case_path: Optional[str] = None 
    
    @classmethod
    def generate_document_id(cls, filename: str) -> str:
        """Generate a unique document ID based on filename and timestamp"""
        timestamp = int(datetime.now().timestamp())
        # Create a safe filename without spaces or special characters
        safe_name = ''.join(c for c in filename.split('.')[0].replace(' ', '_') if c.isalnum() or c == '_')
        return f"doc_{timestamp}_{safe_name}"


class DocumentProcessRequest(BaseModel):
    """Request to process a new document"""
    document_id: Optional[str] = None
    original_filename: str
    metadata: Optional[Dict[str, Any]] = None
    
    @model_validator(mode='after')
    def ensure_document_id(self) -> 'DocumentProcessRequest':
        """Ensure document_id is set, generating one if not provided"""
        if not self.document_id:
            self.document_id = DocumentMetadata.generate_document_id(self.original_filename)
        return self


class DocumentProcessResponse(BaseModel):
    """Response after document processing request is submitted"""
    status: str = "success"
    job_id: str
    documents: List[DocumentMetadata]
    estimated_processing_time: str


class DocumentListResponse(BaseModel):
    """Response for document listing"""
    documents: List[DocumentMetadata]
    pagination: Dict[str, int] = Field(
        default_factory=lambda: {"total": 0, "limit": 10, "offset": 0}
    )


class DocumentDetailResponse(BaseModel):
    """Detailed document information"""
    document_id: str
    document_name: str
    status: DocumentStatus
    chunks_count: int
    processing_stats: Dict[str, Any]
    content_types: Optional[Dict[str, int]] = None
    raptor_levels: Optional[List[int]] = None
    language: Optional[str] = None
    processing_date: Optional[datetime] = None


class DocumentProcessingStatusResponse(BaseModel):
    """Response for document processing status"""
    job_id: str
    status: str
    progress: int
    documents: List[Dict[str, Any]]
    started_at: datetime


class RegenerateEmbeddingsRequest(BaseModel):
    """Request to regenerate document embeddings"""
    document_ids: List[str]
    regenerate_raptor: bool = True