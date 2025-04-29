# api/models/admin_models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class LogSeverity(str, Enum):
    """Severity level for logs"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditLog(BaseModel):
    """System audit log entry"""
    id: str
    timestamp: datetime
    user_id: str
    action: str
    details: Optional[Dict[str, Any]] = None


class ErrorLog(BaseModel):
    """System error log entry"""
    id: str
    timestamp: datetime
    severity: LogSeverity
    component: str
    message: str
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class UserActivity(BaseModel):
    """User activity statistics"""
    documents_used: int
    queries_today: int
    total_queries: int
    last_active: Optional[datetime] = None


class UserInfo(BaseModel):
    """User information with activity stats"""
    id: str
    email: str
    activity_stats: UserActivity


class SystemStorageStats(BaseModel):
    """System storage statistics"""
    total_gb: float
    used_gb: float
    available_gb: float
    usage_percent: float


class SystemHealthStats(BaseModel):
    """System health statistics"""
    storage_usage: SystemStorageStats
    vector_db_status: str
    processing_queue_length: int
    active_jobs: int


class DocumentStats(BaseModel):
    """Aggregate document statistics"""
    total_documents: int
    total_pages: int
    processed_today: int
    total_chunks: int
    avg_processing_time: float


class QueryStats(BaseModel):
    """Aggregate query statistics"""
    queries_today: int
    total_queries: int
    average_query_time: float
    average_tokens_used: int


class SystemStatsResponse(BaseModel):
    """Response with system-wide statistics"""
    document_stats: DocumentStats
    query_stats: QueryStats
    system_health: SystemHealthStats


class ProcessingStageStats(BaseModel):
    """Statistics for a specific processing stage"""
    average_time: float
    total_processed: int
    success_rate: float
    error_count: int


class ProcessingStatsResponse(BaseModel):
    """Response with detailed processing statistics"""
    mineru_stats: ProcessingStageStats
    chunking_stats: ProcessingStageStats
    embedding_stats: ProcessingStageStats
    raptor_stats: ProcessingStageStats


class RetrievalMethodStats(BaseModel):
    """Statistics for a retrieval method"""
    count: int
    average_time: float
    average_token_usage: int


class QueryPerformanceResponse(BaseModel):
    """Response with query performance statistics"""
    overall_stats: Dict[str, Any]
    retrieval_stats: Dict[str, RetrievalMethodStats]


class AuditLogResponse(BaseModel):
    """Response with audit logs"""
    logs: List[AuditLog]
    pagination: Dict[str, int]


class ErrorLogResponse(BaseModel):
    """Response with error logs"""
    logs: List[ErrorLog]
    pagination: Dict[str, int]


class UserListResponse(BaseModel):
    """Response with user information"""
    users: List[UserInfo]
    pagination: Dict[str, int]
    
import logging
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class RetrievalParams(BaseModel):
    """Parameters for retrieval operations"""
    query: str
    case_id: str
    document_ids: List[str]
    use_tree: bool = False
    top_k: int = 5
    
    @property
    def retrieval_method(self) -> str:
        """Get the retrieval method name"""
        return "tree" if self.use_tree else "flat"

class ProcessingStats(BaseModel):
    """Statistics for query processing"""
    retrieval_time: float = 0.0
    llm_time: float = 0.0
    total_time: float = 0.0
    token_count: int = 0
    model_used: Optional[str] = None
    retrieval_method: str = "flat"
    chunks_retrieved: int = 0

class ChunkSource(BaseModel):
    """Source information from a retrieved chunk"""
    document_id: str
    content: str
    score: float
    content_type: str = "text"
    page_number: Optional[int] = None
    tree_level: Optional[int] = None
    original_boxes: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryResult(BaseModel):
    """Result of a query operation"""
    status: str
    answer: str
    sources: List[ChunkSource] = Field(default_factory=list)
    stats: ProcessingStats = Field(default_factory=ProcessingStats)
    message: Optional[str] = None

class TreeNode(BaseModel):
    """Node in the RAPTOR tree structure"""
    node_id: str
    content: str
    level: int
    parent_id: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class HistoryItem(BaseModel):
    """Item in chat history"""
    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class ChatHistory(BaseModel):
    """Container for chat history"""
    items: List[HistoryItem] = Field(default_factory=list)
    summary: Optional[str] = None
    
    def format_for_prompt(self) -> str:
        """Format chat history for inclusion in a prompt"""
        if not self.items:
            return ""
            
        if self.summary:
            history = f"Previous conversation summary: {self.summary}\n\n"
        else:
            history = "Previous conversation:\n\n"
            
        for item in self.items:
            role_name = "Human" if item.role == "user" else "Assistant"
            history += f"{role_name}: {item.content}\n\n"
            
        return history