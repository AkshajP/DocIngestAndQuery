# api/models/task_models.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class TaskStatus(str, Enum):
    """Task status enumeration for API responses"""
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PAUSED = "PAUSED"
    RESUMED = "RESUMED"
    CANCELLED = "CANCELLED"
    RETRY = "RETRY"

class ProcessingStage(str, Enum):
    """Processing stage enumeration for API responses"""
    UPLOAD = "upload"
    EXTRACTION = "extraction"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    TREE_BUILDING = "tree_building"
    VECTOR_STORAGE = "vector_storage"
    COMPLETED = "completed"

class WorkerInfo(BaseModel):
    """Worker information model"""
    worker_id: Optional[str] = None
    worker_hostname: Optional[str] = None

class TaskControlCapabilities(BaseModel):
    """Task control capabilities model"""
    can_pause: bool = False
    can_resume: bool = False
    can_cancel: bool = True
    pause_requested: bool = False
    cancel_requested: bool = False

class TaskSummary(BaseModel):
    """Summary information about a document task"""
    document_id: str
    current_stage: ProcessingStage
    task_status: TaskStatus
    percent_complete: int = Field(ge=0, le=100)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    can_pause: bool = False
    can_resume: bool = False
    can_cancel: bool = True

class TaskDetail(BaseModel):
    """Detailed information about a document task"""
    document_id: str
    current_stage: ProcessingStage
    task_status: TaskStatus
    percent_complete: int = Field(ge=0, le=100)
    
    # Control capabilities
    can_pause: bool = False
    can_resume: bool = False
    can_cancel: bool = True
    pause_requested: bool = False
    cancel_requested: bool = False
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Worker information
    worker_info: Optional[WorkerInfo] = None
    celery_task_id: Optional[str] = None
    
    # Processing information
    completed_stages: List[ProcessingStage] = []
    stage_completion_times: Dict[str, str] = {}
    stage_error_details: Dict[str, Any] = {}
    retry_counts: Dict[str, int] = {}
    
    # Checkpoint and progress data
    checkpoint_data: Dict[str, Any] = {}
    stage_metadata: Dict[str, Any] = {}
    error_details: Dict[str, Any] = {}
    
    # Retry information
    retry_count: int = 0
    max_retries: int = 3
    
    # Last update
    last_updated: Optional[str] = None

class TaskStatusResponse(BaseModel):
    """Response model for task status endpoint"""
    document_id: str
    current_stage: ProcessingStage
    task_status: TaskStatus
    percent_complete: int = Field(ge=0, le=100)
    
    # Control capabilities
    can_pause: bool = False
    can_resume: bool = False
    can_cancel: bool = True
    pause_requested: bool = False
    cancel_requested: bool = False
    
    # Processing progress
    completed_stages: List[ProcessingStage] = []
    stage_completion_times: Dict[str, str] = {}
    stage_error_details: Dict[str, Any] = {}
    retry_counts: Dict[str, int] = {}
    last_updated: Optional[str] = None
    
    # Worker and task information
    worker_info: Optional[WorkerInfo] = None
    celery_task_id: Optional[str] = None

class TaskControlResponse(BaseModel):
    """Response model for task control operations"""
    status: str
    message: str
    document_id: str
    action: Optional[str] = None  # "pause", "resume", "cancel"
    new_task_id: Optional[str] = None  # For resume operations

class TaskControlCapabilitiesResponse(BaseModel):
    """Response model for control capabilities endpoint"""
    document_id: str
    can_pause: bool = False
    can_resume: bool = False
    can_cancel: bool = True
    pause_requested: bool = False
    cancel_requested: bool = False
    task_status: TaskStatus
    percent_complete: int = Field(ge=0, le=100)

class TaskListResponse(BaseModel):
    """Response model for listing tasks"""
    tasks: List[TaskSummary]
    total: int
    offset: int = 0
    limit: int = 100

class TaskHistoryEvent(BaseModel):
    """Model for task history events"""
    event_type: str  # "stage_started", "stage_completed", "stage_failed", "paused", "resumed", "cancelled"
    stage: Optional[ProcessingStage] = None
    timestamp: str
    details: Optional[str] = None
    error_message: Optional[str] = None

class TaskHistoryResponse(BaseModel):
    """Response model for task history"""
    document_id: str
    events: List[TaskHistoryEvent]
    total_events: int

class WorkerStatus(BaseModel):
    """Model for worker status information"""
    worker_id: str
    hostname: str
    status: str  # "active", "inactive", "busy"
    active_tasks: int = 0
    last_heartbeat: Optional[datetime] = None

class WorkerListResponse(BaseModel):
    """Response model for worker list"""
    workers: List[WorkerStatus]
    total_workers: int
    active_workers: int