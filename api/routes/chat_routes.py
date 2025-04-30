from fastapi import APIRouter, Depends, HTTPException, Path, Query, BackgroundTasks
from typing import List, Dict, Any, Optional

from api.models.chat_models import (
    ChatCreateRequest, ChatDetailResponse, ChatUpdateRequest, 
    ChatListResponse, ChatHistoryResponse, ChatDocumentsUpdateRequest
)
from api.models.query_models import QueryRequest, QueryResponse, RegenerateResponseRequest
import time 
from services.chat.manager import ChatManager
from services.chat.history import ChatHistoryService
from services.retrieval.query_engine import QueryEngine
from services.feedback.manager import FeedbackManager

# Dependency functions (to be replaced with actual auth middleware)
async def get_current_user():
    # This would normally verify the token and return the user_id
    return "user_test123"

async def get_current_case():
    # This would normally get the current case from the request
    return "case_demo123"

router = APIRouter(prefix="/api/chats", tags=["chats"])

@router.post("", response_model=ChatDetailResponse)
async def create_chat(
    request: ChatCreateRequest, 
    user_id: str = Depends(get_current_user), 
    case_id: str = Depends(get_current_case)
):
    """Create a new chat"""
    chat_service = ChatManager()
    
    document_ids = [doc.document_id for doc in request.loaded_documents] if request.loaded_documents else []
    
    chat = chat_service.create_chat(
        title=request.title,
        user_id=user_id,
        case_id=case_id,
        document_ids=document_ids,
        settings=request.settings if hasattr(request, "settings") else None
    )
    
    # Format response
    return ChatDetailResponse(
        id=chat.chat_id,
        title=chat.title,
        messages_count=0,
        loaded_documents=request.loaded_documents,
        history={"messages": []}
    )

@router.get("", response_model=ChatListResponse)
async def list_chats(
    include_archived: bool = False,
    limit: int = 10,
    offset: int = 0,
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case)
):
    """List all chats for the current user and case"""
    chat_service = ChatManager()
    
    chats, total = chat_service.list_chats(
        user_id=user_id,
        case_id=case_id,
        include_archived=include_archived
    )
    
    # Format response
    return ChatListResponse(
        chats=[
            {
                "id": chat.chat_id,
                "title": chat.title,
                "messages_count": 0,  # We'd need to count messages in a real implementation
                "last_active": chat.updated_at
            }
            for chat in chats
        ],
        pagination={
            "total": total,
            "limit": limit,
            "offset": offset
        }
    )

@router.get("/{chat_id}", response_model=ChatDetailResponse)
async def get_chat(
    chat_id: str = Path(..., description="Chat ID"),
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case)
):
    """Get chat details including recent messages"""
    chat_service = ChatManager()
    history_service = ChatHistoryService()
    
    # Get chat details
    chat = chat_service.get_chat(
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id
    )
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Get recent messages
    messages, total_count = chat_service.get_chat_history(
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id,
        limit=10  # Get recent messages
    )
    
    # Get document list
    # In a real implementation, we'd query for the loaded documents
    # For now, let's just use mock data
    loaded_documents = []
    
    # Format response
    return ChatDetailResponse(
        id=chat.chat_id,
        title=chat.title,
        messages_count=total_count,
        loaded_documents=loaded_documents,
        history={"messages": messages}
    )

@router.patch("/{chat_id}", response_model=ChatDetailResponse)
async def update_chat(
    request: ChatUpdateRequest,
    chat_id: str = Path(..., description="Chat ID"),
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case)
):
    """Update chat properties"""
    chat_service = ChatManager()
    
    # Update chat
    chat = chat_service.update_chat(
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id,
        title=request.title
    )
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Format response (simplified)
    return ChatDetailResponse(
        id=chat.chat_id,
        title=chat.title,
        messages_count=0,
        loaded_documents=[],
        history={"messages": []}
    )

@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str = Path(..., description="Chat ID"),
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case)
):
    """Delete a chat"""
    chat_service = ChatManager()
    
    success = chat_service.delete_chat(
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    return {"status": "success", "message": "Chat deleted"}

@router.get("/{chat_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    chat_id: str = Path(..., description="Chat ID"),
    limit: int = 20,
    offset: int = 0,
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case)
):
    """Get paginated chat history"""
    chat_service = ChatManager()
    
    messages, total = chat_service.get_chat_history(
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id,
        limit=limit
    )
    
    if not messages and total == 0:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    return ChatHistoryResponse(
        messages=messages,
        pagination={
            "total": total,
            "limit": limit,
            "offset": offset
        }
    )

@router.put("/{chat_id}/documents")
async def update_chat_documents(
    request: ChatDocumentsUpdateRequest,
    chat_id: str = Path(..., description="Chat ID"),
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case)
):
    """Update documents loaded in a chat"""
    chat_service = ChatManager()
    
    success = chat_service.update_chat_documents(
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id,
        add_docs=request.add,
        remove_docs=request.remove
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Get updated document list
    # In a real implementation, we'd query for the loaded documents
    # For now, let's just use mock data
    loaded_documents = []
    
    return {"status": "success", "loaded_documents": loaded_documents}

@router.post("/{chat_id}/query", response_model=QueryResponse)
async def submit_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    chat_id: str = Path(..., description="Chat ID"),
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case)
):
    """Submit a query in a chat"""
    chat_service = ChatManager()
    query_service = QueryEngine()
    history_service = ChatHistoryService()
    
    # Check if chat exists
    chat = chat_service.get_chat(
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id
    )
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Get document IDs for this chat
    # In a real implementation, we'd query for the loaded documents
    # For now, let's just use mock data or pass an empty list
    document_ids = ["doc_1234_sample"]  # Replace with actual document IDs from chat
    
    # Get chat history formatted for prompt
    chat_history = chat_service.get_formatted_history(
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id,
        for_prompt=True
    )
    
    # Process query
    result = query_service.query(
        question=request.question,
        case_id=case_id,
        document_ids=document_ids,
        chat_id=chat_id,
        user_id=user_id,
        use_tree=request.use_tree if request.use_tree is not None else False,
        top_k=request.top_k if request.top_k is not None else 5,
        chat_history=chat_history,
        model_preference=request.model if hasattr(request, "model") else None
    )
    
    # Add to chat history in background to avoid blocking response
    background_tasks.add_task(
        history_service.add_interaction,
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id,
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        token_count=result.get("token_count"),
        model_used=result.get("model_used"),
        response_time=int(result.get("time_taken", 0) * 1000)  # Convert to ms
    )
    
    # Format result for API response
    query_response = QueryResponse(
        id=f"msg_{int(time.time())}",  # In real implementation, use message_id
        answer=result["answer"],
        sources=result["sources"],
        processing_stats={
            "time_taken": result.get("time_taken", 0),
            "tokens_used": result.get("token_count", 0),
            "retrieval_time": result.get("retrieval_time", 0),
            "llm_time": result.get("llm_time", 0),
            "method": result.get("method", "flat")
        }
    )
    
    return query_response

@router.get("/{chat_id}/title")
async def generate_chat_title(
    chat_id: str = Path(..., description="Chat ID"),
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case)
):
    """Generate a title for the chat based on its content"""
    chat_service = ChatManager()
    
    title = chat_service.generate_title(
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id
    )
    
    if not title:
        raise HTTPException(status_code=404, detail="Chat not found or not enough content")
    
    return {"title": title}

@router.post("/{chat_id}/messages/{message_id}/regenerate", response_model=QueryResponse)
async def regenerate_response(
    request: RegenerateResponseRequest,
    background_tasks: BackgroundTasks,
    chat_id: str = Path(..., description="Chat ID"),
    message_id: str = Path(..., description="Message ID"),
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case)
):
    """Regenerate a response for a specific message"""
    chat_service = ChatManager()
    query_service = QueryEngine()
    history_service = ChatHistoryService()
    
    # Get the message
    message = history_service.chat_repo.get_message(message_id)
    
    if not message or message.chat_id != chat_id:
        raise HTTPException(status_code=404, detail="Message not found in this chat")
    
    # Find the previous message (the user's question)
    messages, _ = chat_service.get_chat_history(
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id,
        limit=20  # Get enough messages to find the question
    )
    
    # Find the user's question that preceded this message
    question = None
    for i, msg in enumerate(messages):
        if msg["id"] == message_id and i > 0 and msg["role"] == "assistant":
            prev_msg = messages[i-1]
            if prev_msg["role"] == "user":
                question = prev_msg["content"]
                break
    
    if not question:
        raise HTTPException(status_code=400, detail="Cannot find question for this message")
    
    # Get document IDs for this chat
    # In a real implementation, we'd query for the loaded documents
    document_ids = ["doc_1234_sample"]  # Replace with actual document IDs from chat
    
    # Get chat history up to this message
    # In a real implementation, we'd filter history to exclude this Q&A pair
    chat_history = chat_service.get_formatted_history(
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id,
        for_prompt=True
    )
    
    # Process query
    result = query_service.query(
        question=question,
        case_id=case_id,
        document_ids=document_ids,
        chat_id=chat_id,
        user_id=user_id,
        use_tree=request.use_tree if request.use_tree is not None else False,
        top_k=request.top_k if request.top_k is not None else 5,
        chat_history=chat_history,
        model_preference=request.model if hasattr(request, "model") else None
    )
    
    # Update message in background
    background_tasks.add_task(
        history_service.chat_repo.update_message_status,
        message_id=message_id,
        status="completed",
        error_details=None
    )
    
    # Format result for API response
    query_response = QueryResponse(
        id=message_id,
        answer=result["answer"],
        sources=result["sources"],
        processing_stats={
            "time_taken": result.get("time_taken", 0),
            "tokens_used": result.get("token_count", 0),
            "retrieval_time": result.get("retrieval_time", 0),
            "llm_time": result.get("llm_time", 0),
            "method": result.get("method", "flat")
        }
    )
    
    return query_response