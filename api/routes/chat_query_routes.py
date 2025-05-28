# api/routes/chat_query_routes.py

from fastapi import APIRouter, Depends, HTTPException, Path, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import json
import time 
from datetime import datetime

# Import our new models and helpers
from api.models.chat_request_models import QueryContextRequest, RegenerateContextRequest
from api.routes.chat_helpers import (
    ChatContext, get_chat_context, should_auto_generate_title, auto_generate_title_task
)

# Import existing models and services
from api.models.query_models import QueryResponse
from services.retrieval.query_engine import QueryEngine
from services.chat.history import ChatHistoryService

import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/chats", tags=["chat-queries"])

@router.post("/{chat_id}/query")
async def submit_query(
    request: QueryContextRequest,
    background_tasks: BackgroundTasks,
    context: ChatContext = Depends(get_chat_context),
    stream: bool = Query(None, description="Override request stream setting")
):
    """
    Submit a query in a chat with streamlined parameter handling.
    
    All query configuration is managed through chat settings with optional overrides in the request.
    The response can be streamed (default) or returned as a complete response.
    """
    query_service = QueryEngine()
    
    # Get effective settings by merging request overrides with chat settings
    effective_settings = request.merge_with_chat_settings(context.chat_settings)
    
    # Validate document access
    if not context.document_ids:
        raise HTTPException(
            status_code=400, 
            detail="No documents loaded in this chat. Please add documents before querying."
        )
    
    # Get chat history
    chat_history = context.get_chat_history(for_prompt=True)
    
    # Generate message ID
    message_id = f"msg_{int(time.time())}_{context.chat_id[-6:]}"
    
    # Save user's message
    context.history_service.chat_repo.add_message(
        chat_id=context.chat_id,
        user_id=context.user_id,
        case_id=context.case_id,
        role="user",
        content=request.question,
        status="completed"
    )
    
    # Create assistant message placeholder
    assistant_message = context.history_service.chat_repo.add_message(
        chat_id=context.chat_id,
        user_id="system",
        case_id=context.case_id,
        role="assistant",
        content="",
        status="processing",
        model_used=effective_settings.llm_model
    )
    
    message_id = assistant_message.message_id
    
    # Auto-generate title if needed
    if should_auto_generate_title(context.chat):
        background_tasks.add_task(
            auto_generate_title_task, 
            context=context, 
            question=request.question
        )
    
    # Determine streaming preference (request override > query parameter > default)
    should_stream = stream if stream is not None else (request.stream if request.stream is not None else True)
    
    # Handle streaming vs non-streaming
    if should_stream:
        return StreamingResponse(
            _stream_query_response(
                query_service=query_service,
                context=context,
                question=request.question,
                effective_settings=effective_settings,
                chat_history=chat_history,
                message_id=message_id
            ),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming query processing
        result = query_service.query(
            question=request.question,
            case_id=context.case_id,
            document_ids=context.document_ids,
            chat_id=context.chat_id,
            user_id=context.user_id,
            chat_settings=effective_settings,
            chat_history=chat_history
        )
        
        # Update message with results
        background_tasks.add_task(
            _update_message_with_result,
            context.history_service,
            message_id,
            result
        )
        
        return QueryResponse(
            id=message_id,
            answer=result["answer"],
            sources=result["sources"],
            processing_stats=_create_processing_stats(result, effective_settings)
        )

@router.post("/{chat_id}/messages/{message_id}/regenerate")
async def regenerate_response(
    request: RegenerateContextRequest,
    background_tasks: BackgroundTasks,
    message_id: str = Path(..., description="Message ID"),
    context: ChatContext = Depends(get_chat_context),
    stream: bool = Query(None, description="Override request stream setting")
):
    """
    Regenerate response with streamlined settings handling and streaming support.
    
    This endpoint allows regenerating a previous assistant response with different settings.
    The response can be streamed (default) or returned as a complete response.
    
    Features:
    - Streaming support (configurable via query parameter or request body)
    - Settings inheritance from chat with request-level overrides
    - Automatic chat history filtering (excludes the Q&A pair being regenerated)
    - Progress tracking with Server-Sent Events when streaming
    """
    query_service = QueryEngine()
    
    # Get the message to regenerate
    message = context.history_service.chat_repo.get_message(message_id)
    if not message or message.chat_id != context.chat_id:
        raise HTTPException(status_code=404, detail="Message not found in this chat")
    
    # Find the corresponding user question
    question = _find_question_for_message(context, message_id)
    if not question:
        raise HTTPException(status_code=400, detail="Cannot find question for this message")
    
    # Validate document access
    if not context.document_ids:
        raise HTTPException(
            status_code=400, 
            detail="No documents loaded in this chat"
        )
    
    # Get effective settings
    effective_settings = request.merge_with_chat_settings(context.chat_settings)
    
    # Get chat history (filtered to exclude this Q&A pair)
    chat_history = _get_filtered_chat_history(context, message_id)
    
    # Reset message status to processing
    context.history_service.chat_repo.update_message_status(
        message_id=message_id,
        status="processing"
    )
    
    # Determine streaming preference (query param > request > default True)
    should_stream = stream if stream is not None else (getattr(request, 'stream', None) if hasattr(request, 'stream') else True)
    
    # Handle streaming vs non-streaming
    if should_stream:
        return StreamingResponse(
            _stream_regenerate_response(
                query_service=query_service,
                context=context,
                question=question,
                effective_settings=effective_settings,
                chat_history=chat_history,
                message_id=message_id
            ),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming regeneration
        result = query_service.query(
            question=question,
            case_id=context.case_id,
            document_ids=context.document_ids,
            chat_id=context.chat_id,
            user_id=context.user_id,
            chat_settings=effective_settings,
            chat_history=chat_history
        )
        
        # Update message in background
        background_tasks.add_task(
            _update_message_with_result,
            context.history_service,
            message_id,
            result
        )
        
        return QueryResponse(
            id=message_id,
            answer=result["answer"],
            sources=result["sources"],
            processing_stats=_create_processing_stats(result, effective_settings)
        )
        
@router.get("/{chat_id}/messages/{message_id}/highlights/{document_id}")
async def get_message_highlights(
    message_id: str = Path(..., description="Message ID"),
    document_id: str = Path(..., description="Document ID"),
    context: ChatContext = Depends(get_chat_context),
    highlight_data: Optional[str] = Query(None, description="JSON string of highlights data"),
    zoom: float = Query(1.5, description="Zoom level for highlighting")
):
    """Get highlighted PDF regions for message citations with streamlined access control"""
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
        # Get highlights from message sources
        message = context.history_service.chat_repo.get_message(message_id)
        
        if not message or message.chat_id != context.chat_id:
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
        raise HTTPException(
            status_code=404, 
            detail="Could not generate highlights for this document"
        )
    
    return {"image_data": img_str}

# Helper functions
async def _stream_query_response(
    query_service, context: ChatContext, question: str, 
    effective_settings, chat_history: str, message_id: str
) -> AsyncGenerator[str, None]:
    """Streamlined streaming response handler"""
    start_time = time.time()
    
    yield "event: start\n"
    yield f"data: {json.dumps({'message_id': message_id})}\n\n"
    
    try:
        # Get query embedding
        retrieval_start = time.time()
        query_embedding = await asyncio.to_thread(
            query_service.embeddings.embed_query, question
        )
        
        # Retrieve chunks using effective settings
        chunks = await asyncio.to_thread(
            query_service.retrieve_relevant_chunks,
            query_embedding=query_embedding,
            query_text=question,
            case_id=context.case_id,
            document_ids=context.document_ids,
            use_tree=effective_settings.use_tree_search,
            use_hybrid=effective_settings.use_hybrid_search,
            vector_weight=effective_settings.vector_weight,
            top_k=effective_settings.top_k,
            tree_level_filter=effective_settings.tree_level_filter
        )
        
        retrieval_time = time.time() - retrieval_start
        
        yield "event: retrieval_complete\n"
        yield f"data: {json.dumps({'chunks_count': len(chunks), 'time': retrieval_time})}\n\n"
        
        if not chunks:
            yield "event: error\n"
            yield f"data: {json.dumps({'error': 'No relevant information found in the documents'})}\n\n"
            
            # Update message status
            await asyncio.to_thread(
                context.history_service.chat_repo.update_message_status,
                message_id,
                "failed",
                error_details={"error": "No relevant information found"}
            )
            return
        
        # Send sources
        sources = _format_sources_for_streaming(chunks)
        yield "event: sources\n"
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        
        # Stream response
        llm_start = time.time()
        full_answer = ""
        token_count = 0
        
        async for token in query_service.stream_response(
            question=question,
            chunks=chunks,
            chat_history=chat_history,
            model=effective_settings.llm_model
        ):
            full_answer += token
            token_count += 1
            
            yield "event: token\n"
            yield f"data: {json.dumps({'token': token})}\n\n"
            
            # Periodic updates
            if token_count % 10 == 0:
                await asyncio.to_thread(
                    context.history_service.chat_repo.update_message_content,
                    message_id, full_answer
                )
        
        # Final update
        await asyncio.to_thread(
            context.history_service.chat_repo.update_message_content,
            message_id, full_answer
        )
        
        llm_time = time.time() - llm_start
        total_time = time.time() - start_time
        
        # Update final message status
        await asyncio.to_thread(
            context.history_service.chat_repo.update_message_status,
            message_id=message_id,
            status="completed",
            response_time=int(total_time * 1000),
            sources=chunks,
            token_count=token_count,
            model_used=effective_settings.llm_model
        )
        
        # Complete
        yield "event: complete\n"
        data = {
            "message_id": message_id,
            "time_taken": total_time,
            "llm_time": llm_time,
            "retrieval_time": retrieval_time,
            "token_count": token_count
        }
        yield f"data: {json.dumps(data)}\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming query: {str(e)}")
        yield "event: error\n"
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        # Update message with error
        await asyncio.to_thread(
            context.history_service.chat_repo.update_message_status,
            message_id,
            "failed",
            error_details={"error": str(e)}
        )

def _update_message_with_result(history_service, message_id: str, result: dict):
    """Update message with query result"""
    history_service.chat_repo.update_message_status(
        message_id=message_id,
        status="completed",
        response_time=int(result.get("time_taken", 0) * 1000),
        sources=result.get("sources", []),
        token_count=result.get("input_tokens", 0) + result.get("output_tokens", 0),
        model_used=result.get("model_used")
    )

def _create_processing_stats(result: dict, settings) -> dict:
    """Create processing stats from result"""
    return {
        "time_taken": result.get("time_taken", 0),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "total_tokens": result.get("input_tokens", 0) + result.get("output_tokens", 0),
        "retrieval_time": result.get("retrieval_time", 0),
        "llm_time": result.get("llm_time", 0),
        "method": "tree" if settings.use_tree_search else "flat",
        "model_used": result.get("model_used"),
        "search_method": "hybrid" if settings.use_hybrid_search else "vector"
    }

def _find_question_for_message(context: ChatContext, message_id: str) -> Optional[str]:
    """Find the user question that preceded a given message"""
    messages, _ = context.chat_service.get_chat_history(
        chat_id=context.chat_id,
        user_id=context.user_id,
        case_id=context.case_id,
        limit=20
    )
    
    for i, msg in enumerate(messages):
        if msg["id"] == message_id and i > 0 and msg["role"] == "assistant":
            prev_msg = messages[i-1]
            if prev_msg["role"] == "user":
                return prev_msg["content"]
    return None

def _format_sources_for_streaming(chunks: List[dict]) -> List[dict]:
    """Format chunks for streaming sources event"""
    return [{
        "document_id": chunk["document_id"],
        "content": chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"],
        "score": chunk["score"]
    } for chunk in chunks]

async def _stream_regenerate_response(
    query_service, context: ChatContext, question: str, 
    effective_settings, chat_history: str, message_id: str
) -> AsyncGenerator[str, None]:
    """Streaming response handler for regeneration"""
    start_time = time.time()
    
    yield "event: start\n"
    yield f"data: {json.dumps({'message_id': message_id, 'type': 'regenerate'})}\n\n"
    
    try:
        # Get query embedding
        retrieval_start = time.time()
        query_embedding = await asyncio.to_thread(
            query_service.embeddings.embed_query, question
        )
        
        # Retrieve chunks using effective settings
        chunks = await asyncio.to_thread(
            query_service.retrieve_relevant_chunks,
            query_embedding=query_embedding,
            query_text=question,
            case_id=context.case_id,
            document_ids=context.document_ids,
            use_tree=effective_settings.use_tree_search,
            use_hybrid=effective_settings.use_hybrid_search,
            vector_weight=effective_settings.vector_weight,
            top_k=effective_settings.top_k,
            tree_level_filter=effective_settings.tree_level_filter
        )
        
        retrieval_time = time.time() - retrieval_start
        
        yield "event: retrieval_complete\n"
        yield f"data: {json.dumps({'chunks_count': len(chunks), 'time': retrieval_time})}\n\n"
        
        if not chunks:
            yield "event: error\n"
            yield f"data: {json.dumps({'error': 'No relevant information found in the documents'})}\n\n"
            
            # Update message status
            await asyncio.to_thread(
                context.history_service.chat_repo.update_message_status,
                message_id,
                "failed",
                error_details={"error": "No relevant information found"}
            )
            return
        
        # Send sources
        sources = _format_sources_for_streaming(chunks)
        yield "event: sources\n"
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        
        # Clear existing message content before regenerating
        await asyncio.to_thread(
            context.history_service.chat_repo.update_message_content,
            message_id, ""
        )
        
        # Stream response
        llm_start = time.time()
        full_answer = ""
        token_count = 0
        
        async for token in query_service.stream_response(
            question=question,
            chunks=chunks,
            chat_history=chat_history,
            model=effective_settings.llm_model
        ):
            full_answer += token
            token_count += 1
            
            yield "event: token\n"
            yield f"data: {json.dumps({'token': token})}\n\n"
            
            # Periodic updates
            if token_count % 10 == 0:
                await asyncio.to_thread(
                    context.history_service.chat_repo.update_message_content,
                    message_id, full_answer
                )
        
        # Final update
        await asyncio.to_thread(
            context.history_service.chat_repo.update_message_content,
            message_id, full_answer
        )
        
        llm_time = time.time() - llm_start
        total_time = time.time() - start_time
        
        # Update final message status
        await asyncio.to_thread(
            context.history_service.chat_repo.update_message_status,
            message_id=message_id,
            status="completed",
            response_time=int(total_time * 1000),
            sources=chunks,
            token_count=token_count,
            model_used=effective_settings.llm_model
        )
        
        # Complete
        yield "event: complete\n"
        data = {
            "message_id": message_id,
            "time_taken": total_time,
            'type': 'regenerate',
            "llm_time": llm_time,
            "retrieval_time": retrieval_time,
            "token_count": token_count
        }
        yield f"data: {json.dumps(data)}\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming regeneration: {str(e)}")
        yield "event: error\n"
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        # Update message with error
        await asyncio.to_thread(
            context.history_service.chat_repo.update_message_status,
            message_id,
            "failed",
            error_details={"error": str(e)}
        )

def _get_filtered_chat_history(context: ChatContext, exclude_message_id: str) -> str:
    """Get chat history excluding the Q&A pair containing the specified message"""
    try:
        # Get all messages
        messages, _ = context.chat_service.get_chat_history(
            chat_id=context.chat_id,
            user_id=context.user_id,
            case_id=context.case_id,
            limit=50  # Get more messages to find the Q&A pair
        )
        
        # Find the Q&A pair to exclude
        filtered_messages = []
        skip_next_user = False
        
        for i, msg in enumerate(messages):
            # If this is the message to exclude, mark to skip the previous user message too
            if msg["id"] == exclude_message_id and msg["role"] == "assistant":
                # Skip this assistant message and find the corresponding user message
                for j in range(i-1, -1, -1):  # Look backwards
                    if messages[j]["role"] == "user":
                        # Mark this user message to be skipped
                        messages[j]["_skip"] = True
                        break
                continue
            
            # Skip messages marked for exclusion
            if not msg.get("_skip", False):
                filtered_messages.append(msg)
        
        # Format for prompt (reuse the chat service's formatting)
        if not filtered_messages:
            return ""
        
        history = ""
        for msg in filtered_messages[-10:]:  # Last 10 messages for context
            role_name = "Human" if msg["role"] == "user" else "Assistant"
            history += f"{role_name}: {msg['content']}\n\n"
            
        return history.strip()
        
    except Exception as e:
        logger.warning(f"Error filtering chat history: {str(e)}")
        # Fallback to regular history if filtering fails
        return context.get_chat_history(for_prompt=True)