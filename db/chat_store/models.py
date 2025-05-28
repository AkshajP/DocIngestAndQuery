# db/chat_store/models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class MessageStatus(str, Enum):
    """Status of a message in processing pipeline"""
    SENT = "sent"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class MessageRole(str, Enum):
    """Role of message sender"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatState(str, Enum):
    """State of a chat"""
    OPEN = "open"
    ARCHIVED = "archived"
    DELETED = "deleted"

class UserRole(str, Enum):
    """Role of a user in a case"""
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"

class Message(BaseModel):
    """Individual chat message"""
    message_id: str
    chat_id: str
    user_id: str
    case_id: str
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: MessageStatus = MessageStatus.SENT
    token_count: Optional[int] = None
    model_used: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    response_time: Optional[int] = None

class Feedback(BaseModel):
    """User feedback on a message"""
    id: str
    message_id: str
    user_id: str
    rating: Optional[int] = None
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    feedback_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatSettings(BaseModel):
    """Chat-specific query settings"""
    # Search method settings
    use_tree_search: bool = Field(default=False, description="Use hierarchical tree search")
    use_hybrid_search: bool = Field(default=True, description="Use hybrid (vector + BM25) search")
    vector_weight: float = Field(default=0.65, min=0.0, max=1.0, description="Weight for vector scores in hybrid search")
    
    # Retrieval settings
    top_k: int = Field(default=10, ge=1, le=50, description="Number of chunks to retrieve")
    tree_level_filter: Optional[List[int]] = Field(default=None, description="Filter by tree levels (None = all levels)")
    content_types: Optional[List[str]] = Field(default=None, description="Filter by content types")
    
    # Model settings
    llm_model: Optional[str] = Field(default='llama3.2', description="Override LLM model")
    
    # UI preferences
    show_sources: bool = Field(default=True, description="Show sources by default")
    auto_scroll: bool = Field(default=True, description="Auto-scroll to new messages")
    
    class Config:
        schema_extra = {
            "example": {
                "use_tree_search": False,
                "use_hybrid_search": True,
                "vector_weight": 0.4,
                "top_k": 15,
                "tree_level_filter": [0],  # Only original chunks
                "llm_model": "llama3.2",
                "show_sources": True,
                "auto_scroll": True
            }
        }

class Chat(BaseModel):
    """Chat session information"""
    chat_id: str
    title: str
    user_id: str
    case_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    state: ChatState = ChatState.OPEN
    settings: ChatSettings = Field(default_factory=ChatSettings)
    
class ChatDocument(BaseModel):
    """Document loaded in a chat"""
    id: Optional[int] = None
    chat_id: str
    document_id: str
    added_at: datetime = Field(default_factory=datetime.now)

class UserCaseAccess(BaseModel):
    """User access to a case"""
    id: Optional[int] = None
    user_id: str
    case_id: str
    role: UserRole
    created_at: datetime = Field(default_factory=datetime.now)

