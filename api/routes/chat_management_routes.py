# api/routes/chat_management_routes.py

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Import our new models and helpers
from api.routes.chat_helpers import (
    ChatContext, get_chat_context, create_chat_detail_response
)

# Import existing models and services
from db.chat_store.models import ChatSettings
from api.models.chat_models import (
    ChatCreateRequest, ChatDetailResponse, ChatUpdateRequest, 
    ChatListResponse, ChatHistoryResponse, ChatDocumentsUpdateRequest
)
from services.chat.manager import ChatManager
from services.chat.history import ChatHistoryService
from db.document_store.repository import DocumentMetadataRepository
from api.routes.access_control import get_current_user, get_current_case, validate_user_case_access

import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/chats", tags=["chat-management"])

@router.post("", response_model=ChatDetailResponse)
async def create_chat(
    request: ChatCreateRequest, 
    user_id: str = Depends(get_current_user), 
    case_id: str = Depends(get_current_case),
    _: bool = Depends(validate_user_case_access)
):
    """
    Create a new chat with welcome message.
    
    If the title is not provided or is set to "New Chat" or "Untitled Chat",
    the system will automatically generate a title based on the first user message.
    """
    chat_service = ChatManager()
    history_service = ChatHistoryService()
    
    # Extract document IDs from loaded_documents
    document_ids = [doc.document_id for doc in request.loaded_documents] if request.loaded_documents else []
    
    # Use the provided title or generate a default one
    title = request.title if request.title else "New Chat"
    
    # Create chat with settings
    chat = chat_service.create_chat(
        title=title,
        user_id=user_id,
        case_id=case_id,
        document_ids=document_ids,
        settings=request.settings.dict() if request.settings else None
    )
    
    # Determine welcome message based on loaded documents
    if document_ids:
        welcome_message = "Hello! I'm your document assistant. Ask me questions about the documents you've loaded."
    else:
        welcome_message = "Hello! I'm your document assistant. Please load documents using the sidebar before asking questions."
    
    # Add welcome message to history
    history_service.add_interaction(
        chat_id=chat.chat_id,
        user_id="system",
        case_id=case_id,
        question="",  # No user question for welcome message
        answer=welcome_message,
        sources=None,
        token_count=None,
        model_used=None
    )
    
    # Create context for response generation
    context = ChatContext(chat_id=chat.chat_id, user_id=user_id, case_id=case_id)
    
    return create_chat_detail_response(context, include_history=True, history_limit=1)

@router.get("", response_model=ChatListResponse)
async def list_chats(
    include_archived: bool = Query(False, description="Include archived chats"),
    limit: int = Query(10, ge=1, le=100, description="Number of chats per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case),
    _: bool = Depends(validate_user_case_access)
):
    """List all chats for the current user and case with improved error handling"""
    chat_service = ChatManager()
    
    try:
        chats, total = chat_service.list_chats(
            user_id=user_id,
            case_id=case_id,
            include_archived=include_archived
        )
        
        # Apply pagination
        paginated_chats = chats[offset:offset+limit]
        
        # Format response with actual message counts
        chat_list = []
        for chat in paginated_chats:
            try:
                # Get message count for this chat
                _, count = chat_service.get_chat_history(
                    chat_id=chat.chat_id,
                    user_id=user_id,
                    case_id=case_id,
                    limit=0  # Only need the count, not the messages
                )
                
                # Ensure count is an integer
                count = count if isinstance(count, int) else 0
                    
                chat_list.append({
                    "id": chat.chat_id,
                    "title": chat.title,
                    "messages_count": count,
                    "last_active": chat.updated_at
                })
            except Exception as e:
                logger.warning(f"Error getting message count for chat {chat.chat_id}: {str(e)}")
                # Use default values if anything goes wrong
                chat_list.append({
                    "id": chat.chat_id,
                    "title": chat.title,
                    "messages_count": 0,
                    "last_active": chat.updated_at
                })
        
        return ChatListResponse(
            chats=chat_list,
            pagination={
                "total": total,
                "limit": limit,
                "offset": offset
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing chats for user {user_id}, case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chats")

@router.get("/{chat_id}", response_model=ChatDetailResponse)
async def get_chat(context: ChatContext = Depends(get_chat_context)):
    """Get chat details including recent messages with streamlined context handling"""
    return create_chat_detail_response(context, include_history=True, history_limit=10)

@router.patch("/{chat_id}")
async def update_chat(
    request: ChatUpdateRequest,
    context: ChatContext = Depends(get_chat_context)
):
    """Update chat properties with improved validation"""
    try:
        # Prepare update parameters
        update_params = {}
        if request.title is not None:
            update_params['title'] = request.title
        if request.settings is not None:
            update_params['settings'] = request.settings.dict()
        
        # Perform update
        updated_chat = context.chat_service.update_chat(
            chat_id=context.chat_id,
            user_id=context.user_id,
            case_id=context.case_id,
            **update_params
        )
        
        if not updated_chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Clear cached data in context
        context._chat = updated_chat
        context._chat_settings = None
        
        return create_chat_detail_response(context, include_history=False)
        
    except Exception as e:
        logger.error(f"Error updating chat {context.chat_id}: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail="Failed to update chat")

@router.delete("/{chat_id}")
async def delete_chat(context: ChatContext = Depends(get_chat_context)):
    """Delete a chat with confirmation"""
    try:
        success = context.chat_service.delete_chat(
            chat_id=context.chat_id,
            user_id=context.user_id,
            case_id=context.case_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        return {"status": "success", "message": f"Chat {context.chat_id} deleted"}
        
    except Exception as e:
        logger.error(f"Error deleting chat {context.chat_id}: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail="Failed to delete chat")

@router.get("/{chat_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    context: ChatContext = Depends(get_chat_context),
    limit: int = Query(20, ge=1, le=100, description="Number of messages per page"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """Get paginated chat history with improved pagination"""
    try:
        messages, total = context.chat_service.get_chat_history(
            chat_id=context.chat_id,
            user_id=context.user_id,
            case_id=context.case_id,
            limit=limit
        )
        
        # Apply offset if needed (in case repository doesn't handle it)
        if offset > 0:
            messages = messages[offset:]
        
        return ChatHistoryResponse(
            messages=messages,
            pagination={
                "total": total,
                "limit": limit,
                "offset": offset
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting chat history for {context.chat_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@router.put("/{chat_id}/documents")
async def update_chat_documents(
    request: ChatDocumentsUpdateRequest,
    context: ChatContext = Depends(get_chat_context)
):
    """Update documents loaded in a chat with validation"""
    try:
        # Validate input
        add_docs = request.add if request.add else []
        remove_docs = request.remove if request.remove else []
        
        if not add_docs and not remove_docs:
            raise HTTPException(status_code=400, detail="No documents specified to add or remove")
        
        # Perform update
        success = context.chat_service.update_chat_documents(
            chat_id=context.chat_id,
            user_id=context.user_id,
            case_id=context.case_id,
            add_docs=add_docs,
            remove_docs=remove_docs
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Clear cached document IDs
        context._document_ids = None
        
        # Get updated document list
        loaded_documents = context.get_loaded_documents_info()
        
        return {
            "status": "success", 
            "loaded_documents": loaded_documents,
            "message": f"Updated documents: +{len(add_docs)} -{len(remove_docs)}"
        }
        
    except Exception as e:
        logger.error(f"Error updating documents for chat {context.chat_id}: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail="Failed to update chat documents")

@router.get("/{chat_id}/title")
async def generate_chat_title(context: ChatContext = Depends(get_chat_context)):
    """Generate a title for the chat based on its content"""
    try:
        title = context.chat_service.generate_title(
            chat_id=context.chat_id,
            user_id=context.user_id,
            case_id=context.case_id
        )
        
        if not title:
            raise HTTPException(
                status_code=400, 
                detail="Not enough content to generate title"
            )
        
        # Update the chat with the generated title
        updated_chat = context.chat_service.update_chat(
            chat_id=context.chat_id,
            user_id=context.user_id,
            case_id=context.case_id,
            title=title
        )
        
        if not updated_chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        return {"title": title, "status": "success"}
        
    except Exception as e:
        logger.error(f"Error generating title for chat {context.chat_id}: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail="Failed to generate chat title")

@router.put("/{chat_id}/settings")
async def update_chat_settings(
    settings: ChatSettings,
    context: ChatContext = Depends(get_chat_context)
):
    """Update chat settings with validation"""
    try:
        # Validate settings by attempting to create ChatSettings object
        validated_settings = ChatSettings(**settings.dict())
        
        # Update chat settings
        context.update_settings(validated_settings)
        
        return {
            "status": "success", 
            "settings": validated_settings,
            "message": "Chat settings updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error updating settings for chat {context.chat_id}: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=400, detail=f"Invalid settings: {str(e)}")

@router.get("/{chat_id}/settings")
async def get_chat_settings(context: ChatContext = Depends(get_chat_context)):
    """Get current chat settings"""
    return {
        "settings": context.chat_settings,
        "status": "success"
    }

@router.post("/{chat_id}/archive")
async def archive_chat(context: ChatContext = Depends(get_chat_context)):
    """Archive a chat (soft delete)"""
    try:
        archived_chat = context.chat_service.archive_chat(
            chat_id=context.chat_id,
            user_id=context.user_id,
            case_id=context.case_id
        )
        
        if not archived_chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        return {
            "status": "success", 
            "message": f"Chat '{archived_chat.title}' archived"
        }
        
    except Exception as e:
        logger.error(f"Error archiving chat {context.chat_id}: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail="Failed to archive chat")

# Document management endpoints (global scope)
@router.get("/all-documents")
async def list_processed_documents(
    limit: int = Query(10, ge=1, le=100, description="Number of documents per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    status_filter: Optional[str] = Query(None, description="Filter by document status"),
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case),
    _: bool = Depends(validate_user_case_access)
):
    """List all processed documents accessible to the current user"""
    try:
        doc_repository = DocumentMetadataRepository()
        
        # Get all documents for the case
        all_documents = doc_repository.list_documents_by_case(case_id=case_id)
        
        # Apply status filter if provided
        if status_filter:
            all_documents = [
                doc for doc in all_documents 
                if doc.get("status", "").lower() == status_filter.lower()
            ]
        
        # Apply pagination
        total = len(all_documents)
        paginated_documents = all_documents[offset:offset+limit]
        
        # Format response
        formatted_documents = []
        for doc in paginated_documents:
            formatted_documents.append({
                "document_id": doc.get("document_id", ""),
                "document_name": doc.get("original_filename", "Unnamed"),
                "status": doc.get("status", "Unknown"),
                "chunks_count": doc.get("chunks_count", 0),
                "processing_date": doc.get("processing_date"),
                "file_type": doc.get("file_type"),
                "page_count": doc.get("page_count")
            })
        
        return {
            "documents": formatted_documents,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset
            },
            "filters": {
                "status": status_filter
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing documents for case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@router.get("/stats")
async def get_chat_stats(
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case),
    _: bool = Depends(validate_user_case_access)
):
    """Get chat statistics for the current user and case"""
    try:
        chat_service = ChatManager()
        
        # Get all chats
        all_chats, total_chats = chat_service.list_chats(
            user_id=user_id,
            case_id=case_id,
            include_archived=True
        )
        
        # Calculate statistics
        active_chats = len([chat for chat in all_chats if chat.state == "open"])
        archived_chats = len([chat for chat in all_chats if chat.state == "archived"])
        
        # Get total messages count (approximate)
        total_messages = 0
        for chat in all_chats[:10]:  # Sample first 10 chats to avoid performance issues
            try:
                _, count = chat_service.get_chat_history(
                    chat_id=chat.chat_id,
                    user_id=user_id,
                    case_id=case_id,
                    limit=0
                )
                total_messages += count
            except Exception:
                continue
        
        return {
            "total_chats": total_chats,
            "active_chats": active_chats,
            "archived_chats": archived_chats,
            "estimated_total_messages": total_messages,
            "case_id": case_id
        }
        
    except Exception as e:
        logger.error(f"Error getting chat stats for user {user_id}, case {case_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat statistics")