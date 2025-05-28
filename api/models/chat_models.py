# api/models/chat_models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
from db.chat_store.models import ChatSettings

class MessageRole(str, Enum):
    """Role of a message sender"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Individual chat message"""
    id: str = Field(..., description="Unique message identifier")
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    sources: Optional[List[Dict[str, Any]]] = None
    processing_stats: Optional[Dict[str, Any]] = None
    token_count: Optional[int] = None  # Total tokens used
    input_tokens: Optional[int] = None  # Input tokens
    output_tokens: Optional[int] = None  # Output tokens
    model_used: Optional[str] = None  # Model used for this message
    response_time: Optional[int] = None  # Response time in ms


class ChatDocument(BaseModel):
    """Document loaded in a chat"""
    document_id: str
    title: str


class Chat(BaseModel):
    """Chat session information"""
    id: str = Field(..., description="Unique chat identifier")
    title: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    messages_count: int = 0
    loaded_documents: List[ChatDocument] = []


class ChatSummary(BaseModel):
    """Summary information about a chat for listings"""
    id: str
    title: str
    messages_count: int
    last_active: Optional[datetime] = None


class ChatCreateRequest(BaseModel):
    """Request to create a new chat"""
    loaded_documents: Optional[List[ChatDocument]] = []
    title: Optional[str] = "Untitled Chat"
    settings: Optional[ChatSettings] = None

class ChatListResponse(BaseModel):
    """Response for listing chats"""
    chats: List[ChatSummary]
    pagination: Dict[str, int] = Field(
        default_factory=lambda: {"total": 0, "limit": 10, "offset": 0}
    )


class ChatDetailResponse(BaseModel):
    """Response with chat details"""
    id: str
    title: str
    messages_count: int
    loaded_documents: List[ChatDocument]
    history: Dict[str, Any]  # Contains recent messages
    settings: Optional[ChatSettings] = None


class ChatUpdateRequest(BaseModel):
    """Request to update chat properties"""
    title: Optional[str] = None
    settings: Optional[ChatSettings] = None


class ChatDocumentsUpdateRequest(BaseModel):
    """Request to update documents for a chat"""
    add: Optional[List[str]] = []
    remove: Optional[List[str]] = []


class ChatHistoryResponse(BaseModel):
    """Response with chat history"""
    messages: List[Message]
    pagination: Dict[str, int]

class ChatSettingsUpdateRequest(BaseModel):
    """Request to update chat settings"""
    settings: ChatSettings