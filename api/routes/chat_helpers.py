# api/routes/chat_helpers.py

from fastapi import HTTPException, Depends
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

from db.chat_store.models import Chat, ChatSettings
from services.chat.manager import ChatManager
from services.chat.history import ChatHistoryService
from db.document_store.repository import DocumentMetadataRepository
from api.routes.access_control import get_current_user, get_current_case, validate_user_case_access

logger = logging.getLogger(__name__)

class ChatContext:
    """Context object that holds chat-related information and services"""
    
    def __init__(self, chat_id: str, user_id: str, case_id: str):
        self.chat_id = chat_id
        self.user_id = user_id
        self.case_id = case_id
        self.chat_service = ChatManager()
        self.history_service = ChatHistoryService()
        self.doc_repository = DocumentMetadataRepository()
        self._chat = None
        self._chat_settings = None
        self._document_ids = None
    
    @property
    def chat(self) -> Chat:
        """Lazy load and cache chat object"""
        if self._chat is None:
            self._chat = self.chat_service.get_chat(
                chat_id=self.chat_id,
                user_id=self.user_id,
                case_id=self.case_id
            )
            if not self._chat:
                raise HTTPException(status_code=404, detail="Chat not found")
        return self._chat
    
    @property
    def chat_settings(self) -> ChatSettings:
        """Get parsed and validated chat settings"""
        if self._chat_settings is None:
            self._chat_settings = self._parse_chat_settings(self.chat.settings)
        return self._chat_settings
    
    @property
    def document_ids(self) -> List[str]:
        """Get document IDs associated with this chat"""
        if self._document_ids is None:
            self._document_ids = self.chat_service.get_chat_documents(
                chat_id=self.chat_id,
                user_id=self.user_id,
                case_id=self.case_id
            )
        return self._document_ids
    
    def get_chat_history(self, for_prompt: bool = True, limit: int = 10) -> str:
        """Get formatted chat history"""
        return self.chat_service.get_formatted_history(
            chat_id=self.chat_id,
            user_id=self.user_id,
            case_id=self.case_id,
            for_prompt=for_prompt
        )
    
    def get_loaded_documents_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about loaded documents"""
        loaded_documents = []
        for doc_id in self.document_ids:
            doc_metadata = self.doc_repository.get_document(doc_id)
            if doc_metadata:
                loaded_documents.append({
                    "document_id": doc_id,
                    "title": doc_metadata.get("original_filename", "Unnamed Document")
                })
        return loaded_documents
    
    def update_settings(self, new_settings: ChatSettings) -> Chat:
        """Update chat settings"""
        updated_chat = self.chat_service.update_chat(
            chat_id=self.chat_id,
            user_id=self.user_id,
            case_id=self.case_id,
            settings=new_settings.dict()
        )
        # Clear cached settings
        self._chat_settings = None
        self._chat = updated_chat
        return updated_chat
    
    def _parse_chat_settings(self, settings_data: Any) -> ChatSettings:
        """Parse chat settings from various formats"""
        if not settings_data:
            return ChatSettings()
        
        if isinstance(settings_data, dict):
            try:
                return ChatSettings(**settings_data)
            except Exception as e:
                logger.warning(f"Invalid chat settings format: {e}")
                return ChatSettings()
        elif hasattr(settings_data, '__dict__'):
            return settings_data
        else:
            return ChatSettings()

async def get_chat_context(
    chat_id: str,
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case),
    _: bool = Depends(validate_user_case_access)
) -> ChatContext:
    """Dependency that provides validated chat context"""
    return ChatContext(chat_id=chat_id, user_id=user_id, case_id=case_id)

def create_chat_detail_response(
    context: ChatContext,
    include_history: bool = True,
    history_limit: int = 10
) -> Dict[str, Any]:
    """Create a standardized chat detail response"""
    messages = []
    total_count = 0
    
    if include_history:
        messages, total_count = context.chat_service.get_chat_history(
            chat_id=context.chat_id,
            user_id=context.user_id,
            case_id=context.case_id,
            limit=history_limit
        )
    
    return {
        "id": context.chat.chat_id,
        "title": context.chat.title,
        "created_at": context.chat.created_at,
        "updated_at": context.chat.updated_at,
        "messages_count": total_count,
        "loaded_documents": context.get_loaded_documents_info(),
        "history": {"messages": messages},
        "settings": context.chat_settings
    }

def should_auto_generate_title(chat: Chat) -> bool:
    """Check if chat title should be auto-generated"""
    return chat.title in ["New Chat", "Untitled Chat"]

async def auto_generate_title_task(
    context: ChatContext, 
    question: str
) -> None:
    """Background task to auto-generate chat title"""
    try:
        # Generate title from first question
        query_words = question.split()
        if len(query_words) > 5:
            title = " ".join(query_words[:5]) + "..."
        else:
            title = question
            
        # Ensure title isn't too long
        if len(title) > 50:
            title = title[:47] + "..."
            
        # Update chat title
        context.chat_service.update_chat(
            chat_id=context.chat_id,
            user_id=context.user_id,
            case_id=context.case_id,
            title=title
        )
    except Exception as e:
        logger.error(f"Error auto-generating title: {str(e)}")

def validate_document_access(document_ids: List[str], case_id: str) -> List[str]:
    """Validate that documents belong to the case"""
    # This could be expanded with proper document access validation
    # For now, just return the document_ids as-is
    return document_ids