from fastapi import APIRouter, Depends, HTTPException, Path, Query, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import json
import time 
from datetime import datetime

from api.models.chat_models import (
    ChatCreateRequest, ChatDetailResponse, ChatUpdateRequest, 
    ChatListResponse, ChatHistoryResponse, ChatDocumentsUpdateRequest
)
from db.document_store.repository import DocumentMetadataRepository
from services.pdf.highlighter import PDFHighlighter
from api.models.query_models import QueryRequest, QueryResponse, RegenerateResponseRequest
from services.chat.manager import ChatManager
from services.chat.history import ChatHistoryService
from services.retrieval.query_engine import QueryEngine
from services.feedback.manager import FeedbackManager
from core.config import get_config

# Dependency functions (to be replaced with actual auth middleware)
async def get_current_user():
    # This would normally verify the token and return the user_id
    return "user_test"

async def get_current_case():
    # This would normally get the current case from the request
    return "default"

router = APIRouter(prefix="/ai/chats", tags=["chats"])

@router.post("", response_model=ChatDetailResponse)
async def create_chat(
    request: ChatCreateRequest, 
    user_id: str = Depends(get_current_user), 
    case_id: str = Depends(get_current_case)
):
    """
    Create a new chat with welcome message.
    
    If the title is not provided or is set to "New Chat" or "Untitled Chat",
    the system will automatically generate a title based on the first user message.
    """
    chat_service = ChatManager()
    history_service = ChatHistoryService()
    
    document_ids = [doc.document_id for doc in request.loaded_documents] if request.loaded_documents else []
    
    # Use the provided title or generate a default one
    title = request.title if request.title else "New Chat"
    
    chat = chat_service.create_chat(
        title=title,
        user_id=user_id,
        case_id=case_id,
        document_ids=document_ids,
        settings=request.settings if hasattr(request, "settings") else None
    )
    
    # Add an assistant welcome message
    welcome_message = "Hello! I'm your document assistant. Ask me questions about the documents you've loaded."
    
    # If no documents are loaded, adjust the message
    if not document_ids:
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
    
    # Get welcome message to include in response
    messages, _ = chat_service.get_chat_history(
        chat_id=chat.chat_id,
        user_id=user_id,
        case_id=case_id,
        limit=1
    )
    
    # Format response
    return ChatDetailResponse(
        id=chat.chat_id,
        title=title,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        messages_count=1,  # Just the welcome message
        loaded_documents=request.loaded_documents,
        history={"messages": messages}
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
    history_service = ChatHistoryService()
    
    chats, total = chat_service.list_chats(
        user_id=user_id,
        case_id=case_id,
        include_archived=include_archived
    )
    
    # Format response with actual message counts
    chat_list = []
    for chat in chats:
        try:
            # Get message count for this chat - use get_chat_history instead of get_history
            messages, count = chat_service.get_chat_history(
                chat_id=chat.chat_id,
                user_id=user_id,
                case_id=case_id,
                limit=0  # Only need the count, not the messages
            )
            
            # Make sure count is an integer
            if not isinstance(count, int):
                # If count is not an integer, use length of messages or default to 0
                count = len(messages) if messages else 0
                
            chat_list.append({
                "id": chat.chat_id,
                "title": chat.title,
                "messages_count": count,
                "last_active": chat.updated_at
            })
        except Exception as e:
            # If anything goes wrong, use a default count of 0
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

@router.get("/{chat_id}", response_model=ChatDetailResponse)
async def get_chat(
    chat_id: str = Path(..., description="Chat ID"),
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case)
):
    """Get chat details including recent messages"""
    chat_service = ChatManager()
    doc_repository = DocumentMetadataRepository()  # Add this import at the top
    
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
    
    # Get document IDs for this chat
    document_ids = chat_service.get_chat_documents(
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id
    )
    
    # Get document details from repository
    loaded_documents = []
    for doc_id in document_ids:
        doc_metadata = doc_repository.get_document(doc_id)
        if doc_metadata:
            loaded_documents.append({
                "document_id": doc_id,
                "title": doc_metadata.get("original_filename", "Unnamed Document")
            })
    
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
    doc_repository = DocumentMetadataRepository()  # Make sure this is imported at the top
    
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
    document_ids = chat_service.get_chat_documents(
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id
    )
    
    # Get document details from repository
    loaded_documents = []
    for doc_id in document_ids:
        doc_metadata = doc_repository.get_document(doc_id)
        if doc_metadata:
            loaded_documents.append({
                "document_id": doc_id,
                "title": doc_metadata.get("original_filename", "Unnamed Document")
            })
    
    return {"status": "success", "loaded_documents": loaded_documents}

@router.post("/{chat_id}/query")
async def submit_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    chat_id: str = Path(..., description="Chat ID"),
    stream: bool = Query(True, description="Stream the response as Server-Sent Events"),
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case)
):
    """
    Submit a query in a chat with optional streaming.
    
    When stream=True, the response is streamed as Server-Sent Events with the following event types:
    - start: Initial event with message_id
    - retrieval_complete: Sent when relevant chunks are retrieved
    - sources: Provides information about the source documents used
    - token: Each token of the generated response
    - complete: Final event with timing statistics
    - error: Sent if an error occurs during processing
    
    When stream=False, a complete QueryResponse is returned with the full answer and sources.
    """
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
    document_ids = chat_service.get_chat_documents(chat_id)
    if not document_ids:
        # Add placeholder document ID for testing if none found
        document_ids = ["doc_placeholder"]
    
    # Get chat history formatted for prompt
    chat_history = chat_service.get_formatted_history(
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id,
        for_prompt=True
    )
    
    # Generate message ID for this interaction
    message_id = f"msg_{int(time.time())}_{chat_id[-6:]}"
    
    # Save the user's message first
    history_service.chat_repo.add_message(
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id,
        role="user",
        content=request.question,
        status="completed"
    )
    
    # Create assistant message placeholder with 'processing' status
    assistant_message = history_service.chat_repo.add_message(
        chat_id=chat_id,
        user_id="system",
        case_id=case_id,
        role="assistant",
        content="",  # Will be updated with actual content
        status="processing",
        token_count=None,
        model_used=request.model_override
    )
    
    message_id = assistant_message.message_id
    
    # Check if it's the first message and we should auto-generate a title
    if chat.title == "New Chat" or chat.title == "Untitled Chat":
        background_tasks.add_task(
            _auto_generate_title,
            chat_service=chat_service,
            chat_id=chat_id,
            user_id=user_id,
            case_id=case_id,
            question=request.question
        )
    
    # Handle streaming vs non-streaming responses
    if stream:
        return StreamingResponse(
            _stream_query_response(
                query_service=query_service,
                chat_service=chat_service,
                history_service=history_service,
                question=request.question,
                chat_id=chat_id,
                user_id=user_id,
                case_id=case_id,
                document_ids=document_ids,
                use_tree=request.use_tree,
                top_k=request.top_k,
                chat_history=chat_history,
                model_preference=request.model_override,
                message_id=message_id,
                tree_level_filter=request.tree_level_filter
            ),
            media_type="text/event-stream"
        )
    else:
        # Process query (non-streaming)
        result = query_service.query(
            question=request.question,
            case_id=case_id,
            document_ids=document_ids,
            chat_id=chat_id,
            user_id=user_id,
            use_tree=request.use_tree if request.use_tree is not None else False,
            top_k=request.top_k if request.top_k is not None else 10,
            chat_history=chat_history,
            model_override=request.model_override,
            tree_level_filter=request.tree_level_filter
        )
        
        background_tasks.add_task(
            history_service.chat_repo.update_message_content,
            message_id=message_id,
            content=result["answer"]
        )
        
        # Update assistant message with the full response
        background_tasks.add_task(
            history_service.chat_repo.update_message_status,
            message_id=message_id,
            status="completed",
            error_details=None,
            response_time=int(result.get("time_taken", 0) * 1000)  # Convert to ms
        )
        
        # Format result for API response
        query_response = QueryResponse(
            id=message_id,
            answer=result["answer"],
            sources=result["sources"],
            processing_stats={
                "time_taken": result.get("time_taken", 0),
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "total_tokens": result.get("input_tokens", 0) + result.get("output_tokens", 0),
                "retrieval_time": result.get("retrieval_time", 0),
                "llm_time": result.get("llm_time", 0),
                "method": "tree" if request.use_tree else "flat",
                "model_used": result.get("model_used", None)
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
    
    # Update the chat with the generated title
    chat_service.update_chat(
        chat_id=chat_id,
        user_id=user_id,
        case_id=case_id,
        title=title
    )
    
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


async def _stream_query_response(
    query_service, chat_service, history_service,
    question, chat_id, user_id, case_id, document_ids,
    use_tree, top_k, chat_history, model_preference,
    message_id, tree_level_filter
) -> AsyncGenerator[str, None]:
    """Stream the query response"""
    # Start query processing
    start_time = time.time()
    
    # First yield an event indicating query started processing
    yield "event: start\n"
    yield f"data: {json.dumps({'message_id': message_id})}\n\n"
    
    # Get relevant documents first
    retrieval_start = time.time()
    query_embedding = await asyncio.to_thread(
        query_service.embeddings.embed_query,
        question
    )
    # Get document chunks
    chunks = await asyncio.to_thread(
        query_service.retrieve_relevant_chunks,
        query_embedding=query_embedding,  # Use correct parameter name and embedding vector
        case_id=case_id, 
        document_ids=document_ids,
        use_tree=use_tree,
        top_k=top_k
    )
    
    retrieval_time = time.time() - retrieval_start
    
    # Send retrieval event
    yield "event: retrieval_complete\n"
    yield f"data: {json.dumps({'chunks_count': len(chunks), 'time': retrieval_time})}\n\n"
    
    # If no chunks found, send error and stop
    if not chunks:
        yield "event: error\n"
        yield f"data: {json.dumps({'error': 'No relevant information found in the documents'})}\n\n"
        
        # Update message status
        await asyncio.to_thread(
            history_service.chat_repo.update_message_status,
            message_id=message_id,
            status="failed",
            error_details={"error": "No relevant information found"}
        )
        return
    
    # Send source information
    sources = [{
        "document_id": chunk["document_id"],
        "content": chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"],
        "score": chunk["score"]
    } for chunk in chunks]
    
    yield "event: sources\n"
    yield f"data: {json.dumps({'sources': sources})}\n\n"
    
    # Start generating answer
    llm_start = time.time()
    full_answer = ""
    token_count = 0
    
    try:
        # Use streaming generator from query service
        async for token in query_service.stream_response(
            question=question,
            chunks=chunks,
            chat_history=chat_history,
            model=model_preference
        ):
            full_answer += token
            token_count += 1
            
            # Send token to client
            yield "event: token\n"
            yield f"data: {json.dumps({'token': token})}\n\n"
            
            # Periodically update the message content
            if token_count % 10 == 0:
                await asyncio.to_thread(
                    history_service.chat_repo.update_message_content,
                    message_id=message_id,
                    content=full_answer
                )
        
        llm_time = time.time() - llm_start
        total_time = time.time() - start_time
        
        # Update final message
        await asyncio.to_thread(
            history_service.chat_repo.update_message_status,
            message_id=message_id,
            status="completed",
            # content=full_answer,
            # sources=chunks,
            #token_count=token_count,
            #model_used=model_preference,
            response_time=int(total_time * 1000)  # ms
        )
        
        # Send completion event
        yield "event: complete\n"
        yield f"data: {json.dumps({ 'message_id': message_id,'time_taken': total_time,'llm_time': llm_time,'retrieval_time': retrieval_time,'token_count': token_count})}"
        
    except Exception as e:
        # Send error event
        yield "event: error\n"
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        # Update message with error
        await asyncio.to_thread(
            history_service.chat_repo.update_message_status,
            message_id=message_id,
            status="failed",
            error_details={"error": str(e)}
        )

async def _auto_generate_title(
    chat_service, chat_id, user_id, case_id, question
):
    """Automatically generate a title based on the first user query"""
    try:
        # Generate title from first question
        query_words = question.split()
        if len(query_words) > 5:
            # Take first 5 words plus "..."
            title = " ".join(query_words[:5]) + "..."
        else:
            title = question
            
        # Ensure title isn't too long
        if len(title) > 50:
            title = title[:47] + "..."
            
        # Update chat title
        chat_service.update_chat(
            chat_id=chat_id,
            user_id=user_id,
            case_id=case_id,
            title=title
        )
    except Exception as e:
        # Log error but don't fail the request
        print(f"Error auto-generating title: {str(e)}")
        
        
@router.get("/{chat_id}/messages/{message_id}/highlights/{document_id}")
async def get_message_highlights(
    chat_id: str = Path(..., description="Chat ID"),
    message_id: str = Path(..., description="Message ID"),
    document_id: str = Path(..., description="Document ID"),
    highlight_data: Optional[str] = Query(None, description="JSON string of highlights data"),
    zoom: float = Query(1.5, description="Zoom level for highlighting"),
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case)
):
    """Get highlighted PDF regions for message citations"""
    # Initialize the PDF highlighter
    from services.pdf.highlighter import PDFHighlighter
    from core.config import get_config
    
    highlighter = PDFHighlighter(storage_dir=get_config().storage.storage_dir)
    
    # Parse the highlight data if provided
    highlights = []
    if highlight_data:
        try:
            highlights = json.loads(highlight_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid highlight data format")
    else:
        # If no highlight data provided, try to get from message sources
        history_service = ChatHistoryService()
        message = history_service.chat_repo.get_message(message_id)
        
        if not message or message.chat_id != chat_id:
            raise HTTPException(status_code=404, detail="Message not found")
            
        # Extract highlights from message sources
        sources = message.sources or []
        for source in sources:
            if source.get("document_id") == document_id:
                original_boxes = source.get("original_boxes", [])
                if original_boxes:
                    highlights.extend(original_boxes)
    
    # Generate the highlighted thumbnail
    if not highlights:
        raise HTTPException(status_code=400, detail="No highlight data found")
        
    img_str = highlighter.get_multi_highlight_thumbnail(
        document_id=document_id,
        highlights=highlights,
        zoom=zoom
    )
    
    if not img_str:
        raise HTTPException(status_code=404, detail="Could not generate highlights for this document")
    
    return {"image_data": img_str}

@router.get("/all-documents")  # Changed from "/documents" to "/all-documents"
async def list_processed_documents(
    limit: int = Query(10, description="Number of documents per page"),
    offset: int = Query(0, description="Pagination offset"),
    user_id: str = Depends(get_current_user),
    case_id: str = Depends(get_current_case)
):
    """List all processed documents accessible to the current user"""
    doc_repository = DocumentMetadataRepository()
    
    # Get all documents
    all_documents = doc_repository.list_documents()
    
    # Apply pagination
    total = len(all_documents)
    paginated_documents = all_documents[offset:offset+limit]
    
    # Format response
    return {
        "documents": [
            {
                "document_id": doc.get("document_id", ""),
                "document_name": doc.get("original_filename", "Unnamed"),
                "status": doc.get("status", "Unknown"),
                "chunks_count": doc.get("chunks_count", 0),
                "processing_date": doc.get("processing_date")
            }
            for doc in paginated_documents
        ],
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset
        }
    }