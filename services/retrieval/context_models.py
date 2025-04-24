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