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

class Chat(BaseModel):
    """Chat session information"""
    chat_id: str
    title: str
    user_id: str
    case_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    state: ChatState = ChatState.OPEN
    settings: Dict[str, Any] = Field(default_factory=dict)
    
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