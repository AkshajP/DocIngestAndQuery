import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime

from db.chat_store.repository import ChatRepository
from db.chat_store.models import Chat, ChatState, ChatDocument
from db.document_store.repository import DocumentMetadataRepository
from services.chat.history import ChatHistoryService
from core.config import get_config

logger = logging.getLogger(__name__)

class ChatManager:
    """
    Service for managing chat sessions, including document associations
    and chat state transitions.
    """
    
    def __init__(self, config=None):
        """
        Initialize the chat manager.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or get_config()
        self.chat_repo = ChatRepository()
        self.doc_repo = DocumentMetadataRepository()
        self.history_service = ChatHistoryService()
        
        logger.info("Chat manager initialized")
    
    def create_chat(
        self,
        title: str,
        user_id: str,
        case_id: str,
        document_ids: Optional[List[str]] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Chat:
        """
        Create a new chat session.
        
        Args:
            title: Chat title
            user_id: User ID
            case_id: Case ID
            document_ids: Optional list of document IDs to load
            settings: Optional chat settings
            
        Returns:
            Created Chat object
        """
        # Validate document IDs if provided
        if document_ids:
            valid_doc_ids = self._validate_document_ids(document_ids, case_id)
            if len(valid_doc_ids) < len(document_ids):
                logger.warning(f"Some document IDs were invalid for case {case_id}")
                document_ids = valid_doc_ids
        
        # Create chat
        chat = self.chat_repo.create_chat(
            title=title,
            user_id=user_id,
            case_id=case_id,
            document_ids=document_ids,
            settings=settings or {},
            state=ChatState.OPEN
        )
        
        # Add welcome message
        self.history_service.chat_repo.add_message(
            chat_id=chat.chat_id,
            user_id="system",
            case_id=case_id,
            role="system",
            content=f"Welcome to chat: {title}",
            status="completed"
        )
        
        logger.info(f"Created chat {chat.chat_id} for user {user_id} in case {case_id}")
        
        return chat
    
    def get_chat(
        self,
        chat_id: str,
        user_id: Optional[str] = None,
        case_id: Optional[str] = None
    ) -> Optional[Chat]:
        """
        Get chat details.
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID for access control
            case_id: Optional case ID for access control
            
        Returns:
            Chat object or None if not found
        """
        return self.chat_repo.get_chat(chat_id, user_id, case_id)
    
    def update_chat(
        self,
        chat_id: str,
        user_id: Optional[str] = None,
        case_id: Optional[str] = None,
        title: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        state: Optional[str] = None
    ) -> Optional[Chat]:
        """
        Update chat properties.
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID for access control
            case_id: Optional case ID for access control
            title: Optional new title
            settings: Optional new settings
            state: Optional new state
            
        Returns:
            Updated Chat object or None if not found
        """
        return self.chat_repo.update_chat(
            chat_id=chat_id,
            user_id=user_id,
            case_id=case_id,
            title=title,
            settings=settings,
            state=state
        )
    
    def list_chats(
        self,
        user_id: str,
        case_id: Optional[str] = None,
        include_archived: bool = False
    ) -> Tuple[List[Chat], int]:
        """
        List chats for a user.
        
        Args:
            user_id: User ID
            case_id: Optional case ID filter
            include_archived: Whether to include archived chats
            
        Returns:
            Tuple of (list of Chat objects, total count)
        """
        states = [ChatState.OPEN]
        if include_archived:
            states.append(ChatState.ARCHIVED)
            
        # Convert to strings since the repository expects strings
        state_strings = [s.value for s in states]
        
        chats_list = []
        total = 0
        
        # Get chats for each requested state
        for state in state_strings:
            chats, count = self.chat_repo.list_chats(
                user_id=user_id,
                case_id=case_id,
                state=state
            )
            
            chats_list.extend(chats)
            total += count
        
        return chats_list, total
    
    def delete_chat(
        self,
        chat_id: str,
        user_id: Optional[str] = None,
        case_id: Optional[str] = None
    ) -> bool:
        """
        Delete a chat (permanent deletion).
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID for access control
            case_id: Optional case ID for access control
            
        Returns:
            True if deleted
        """
        return self.chat_repo.delete_chat(chat_id, user_id, case_id)
    
    def archive_chat(
        self,
        chat_id: str,
        user_id: Optional[str] = None,
        case_id: Optional[str] = None
    ) -> Optional[Chat]:
        """
        Archive a chat (soft deletion).
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID for access control
            case_id: Optional case ID for access control
            
        Returns:
            Updated Chat object or None if not found
        """
        return self.update_chat(
            chat_id=chat_id,
            user_id=user_id,
            case_id=case_id,
            state=ChatState.ARCHIVED
        )
    
    def update_chat_documents(
        self,
        chat_id: str,
        user_id: Optional[str] = None,
        case_id: Optional[str] = None,
        add_docs: Optional[List[str]] = None,
        remove_docs: Optional[List[str]] = None
    ) -> bool:
        """
        Update documents associated with a chat.
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID for access control
            case_id: Optional case ID for access control
            add_docs: Optional list of document IDs to add
            remove_docs: Optional list of document IDs to remove
            
        Returns:
            True if successful
        """
        # Validate document IDs to add if provided
        if add_docs and case_id:
            valid_doc_ids = self._validate_document_ids(add_docs, case_id)
            if len(valid_doc_ids) < len(add_docs):
                logger.warning(f"Some document IDs were invalid for case {case_id}")
                add_docs = valid_doc_ids
        
        # Update documents
        success = self.chat_repo.update_chat_documents(
            chat_id=chat_id,
            user_id=user_id,
            case_id=case_id,
            add_docs=add_docs,
            remove_docs=remove_docs
        )
        
        if success:
            logger.info(f"Updated documents for chat {chat_id}")
            
            # Record document changes in chat history
            changes = []
            if add_docs:
                changes.append(f"Added document(s): {', '.join(add_docs)}")
            if remove_docs:
                changes.append(f"Removed document(s): {', '.join(remove_docs)}")
                
            if changes:
                change_message = "Document changes: " + "; ".join(changes)
                self.history_service.chat_repo.add_message(
                    chat_id=chat_id,
                    user_id=user_id or "system",
                    case_id=case_id or "system",
                    role="system",
                    content=change_message,
                    status="completed"
                )
        
        return success
    
    def get_chat_history(
        self,
        chat_id: str,
        user_id: Optional[str] = None,
        case_id: Optional[str] = None,
        limit: int = 20
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get chat history with metadata.
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID for access control
            case_id: Optional case ID for access control
            limit: Maximum number of messages
            
        Returns:
            Tuple of (list of message dictionaries, total count)
        """
        messages, total = self.chat_repo.get_messages(
            chat_id=chat_id,
            user_id=user_id,
            case_id=case_id,
            limit=limit,
            offset=0
        )
        
        # Convert to dictionaries with additional metadata
        message_dicts = []
        for msg in messages:
            message_dict = {
                "id": msg.message_id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
                "status": msg.status
            }
            
            # Add sources if available
            if msg.sources:
                message_dict["sources"] = msg.sources
                
            # Add performance metadata if available
            if msg.token_count:
                message_dict["token_count"] = msg.token_count
                
            if msg.model_used:
                message_dict["model_used"] = msg.model_used
                
            message_dicts.append(message_dict)
        
        return message_dicts, total
    
    def get_formatted_history(
        self,
        chat_id: str,
        user_id: Optional[str] = None,
        case_id: Optional[str] = None,
        for_prompt: bool = True
    ) -> str:
        """
        Get formatted history text for inclusion in prompts.
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID for access control
            case_id: Optional case ID for access control
            for_prompt: Whether to format for inclusion in a prompt
            
        Returns:
            Formatted history text
        """
        # Get chat history
        history = self.history_service.get_history(
            chat_id=chat_id,
            user_id=user_id,
            case_id=case_id,
            limit=10  # Limit to recent messages
        )
        
        # Format for prompt or display
        if for_prompt:
            return history.format_for_prompt()
        else:
            # Format for display (less structured)
            formatted = ""
            for item in history.items:
                role_name = "Human" if item.role == "user" else "Assistant"
                formatted += f"{role_name}: {item.content}\n\n"
            return formatted
    
    def generate_title(
        self,
        chat_id: str,
        user_id: Optional[str] = None,
        case_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate a title for a chat based on its content.
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID for access control
            case_id: Optional case ID for access control
            
        Returns:
            Generated title or None if not enough content
        """
        # Get first user message
        messages, _ = self.chat_repo.get_messages(
            chat_id=chat_id,
            user_id=user_id,
            case_id=case_id,
            limit=5,
            offset=0
        )
        
        user_messages = [m for m in messages if m.role == "user"]
        if not user_messages:
            return "New Chat"
        
        # Use first user message for title
        first_message = user_messages[0].content
        
        # Simple title generation - take first few words
        words = first_message.split()
        if len(words) <= 15:
            title = first_message
        else:
            title = " ".join(words[:15]) + "..."
            
        # Ensure title isn't too long
        if len(title) > 100:
            title = title[:97] + "..."
            
        return title
    
    def _validate_document_ids(
        self,
        document_ids: List[str],
        case_id: str
    ) -> List[str]:
        """
        Validate document IDs for a case.
        
        Args:
            document_ids: List of document IDs to validate
            case_id: Case ID
            
        Returns:
            List of valid document IDs
        """
        valid_ids = []
        
        for doc_id in document_ids:
            # Get document metadata
            doc = self.doc_repo.get_document(doc_id)
            
            # Check if document exists and belongs to the case
            if doc and doc.get("case_id") == case_id:
                valid_ids.append(doc_id)
            else:
                logger.warning(f"Document {doc_id} not found or not in case {case_id}")
        
        return valid_ids