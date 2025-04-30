# tests/test_query_engine.py
"""
Test script to verify that the Query Engine and related components work correctly.
"""

import sys
import os
import logging
import time
from typing import List, Dict, Any

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from services.retrieval.query_engine import QueryEngine
from services.chat.history import ChatHistoryService
from services.chat.manager import ChatManager
from services.feedback.manager import FeedbackManager
from core.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_query_engine")

def test_query_engine():
    """Test the Query Engine functionality"""
    
    print("\n=== Testing Query Engine ===")
    
    # Initialize query engine
    query_engine = QueryEngine()
    
    # Get document list from repository
    config = get_config()
    document_ids = ["doc_1745921839_output", ]  # Replace with actual document IDs
    case_id = "default"  # Replace with actual case ID
    
    # Test flat retrieval
    question = "What are the key concepts?"
    
    print(f"\nTesting query: '{question}'")
    print("Using flat retrieval method...")
    
    result = query_engine.query(
        question=question,
        case_id=case_id,
        document_ids=document_ids,
        use_tree=False,
        top_k=5
    )
    
    print(f"\nQuery Status: {result['status']}")
    print(f"Processing Time: {result['time_taken']:.2f}s")
    print(f"Answer: {result['answer'][:100]}..." if len(result['answer']) > 100 else result['answer'])
    print(f"Sources: {len(result['sources'])}")
    print(f"Input Tokens: {result.get('input_tokens', 'N/A')}")
    print(f"Output Tokens: {result.get('output_tokens', 'N/A')}")
    print(f"Model Used: {result.get('model_used', 'N/A')}")
    
    # Test tree-based retrieval
    print("\nTesting tree-based retrieval...")
    
    result = query_engine.query(
        question=question,
        case_id=case_id,
        document_ids=document_ids,
        use_tree=True,
        top_k=5
    )
    
    print(f"\nQuery Status: {result['status']}")
    print(f"Processing Time: {result['time_taken']:.2f}s")
    print(f"Answer: {result['answer'][:100]}..." if len(result['answer']) > 100 else result['answer'])
    print(f"Sources: {len(result['sources'])}")
    
    # Test filtering for original chunks only
    print("\nTesting retrieval with original chunks only...")
    
    result_original = query_engine.query(
        question=question,
        case_id=case_id,
        document_ids=document_ids,
        use_tree=False,
        top_k=5,
        tree_level_filter=[0]  # Only original chunks
    )
    
    print(f"\nQuery Status: {result_original['status']}")
    print(f"Sources: {len(result_original['sources'])}")
    original_count = sum(1 for s in result_original['sources'] if s.get('tree_level', 0) == 0)
    summary_count = sum(1 for s in result_original['sources'] if s.get('tree_level', 0) > 0)
    print(f"Original Chunks: {original_count}, Summary Chunks: {summary_count}")
    
    # Test filtering for summary chunks only
    print("\nTesting retrieval with summary chunks only...")
    
    result_summary = query_engine.query(
        question=question,
        case_id=case_id,
        document_ids=document_ids,
        use_tree=False,
        top_k=5,
        tree_level_filter=[1, 2, 3]  # Only summary chunks
    )
    
    print(f"\nQuery Status: {result_summary['status']}")
    print(f"Sources: {len(result_summary['sources'])}")
    original_count = sum(1 for s in result_summary['sources'] if s.get('tree_level', 0) == 0)
    summary_count = sum(1 for s in result_summary['sources'] if s.get('tree_level', 0) > 0)
    print(f"Original Chunks: {original_count}, Summary Chunks: {summary_count}")
    
    # Test model override
    print("\nTesting model override...")
    
    result_model = query_engine.query(
        question=question,
        case_id=case_id,
        document_ids=document_ids,
        use_tree=False,
        top_k=5,
        model_override="mistral"  # Use different model
    )
    
    print(f"\nQuery Status: {result_model['status']}")
    print(f"Model Used: {result_model.get('model_used', 'N/A')}")
    
    return result

def test_chat_flow():
    """Test the complete chat flow with history and feedback"""
    
    print("\n=== Testing Complete Chat Flow ===")
    
    # Initialize components
    chat_manager = ChatManager()
    query_engine = QueryEngine()
    history_service = ChatHistoryService()
    feedback_manager = FeedbackManager()
    
    # Test case and user IDs
    case_id = "default"
    user_id = "user_test"
    document_ids = ["doc_1745921839_output"]  # Replace with actual document IDs
    
    # 1. Create a new chat
    print("\n1. Creating a new chat...")
    chat = chat_manager.create_chat(
        title="Test Chat",
        user_id=user_id,
        case_id=case_id,
        document_ids=document_ids
    )
    
    print(f"Created chat: {chat.chat_id}")
    
    # 2. Send a query
    print("\n2. Sending first query...")
    question = "What is the purpose of the document chunking process?"
    
    # Get formatted history
    chat_history = chat_manager.get_formatted_history(
        chat_id=chat.chat_id,
        user_id=user_id,
        case_id=case_id
    )
    
    # Process query
    result = query_engine.query(
        question=question,
        case_id=case_id,
        document_ids=document_ids,
        use_tree=False,
        top_k=3,
        chat_history=chat_history
    )
    
    # Print token information
    print(f"Input Tokens: {result.get('input_tokens', 'N/A')}")
    print(f"Output Tokens: {result.get('output_tokens', 'N/A')}")
    
    # 3. Add interaction to history
    print("\n3. Adding interaction to history...")
    # Ensure token count is an integer
    input_tokens = int(result.get('input_tokens', 0))
    output_tokens = int(result.get('output_tokens', 0))
    total_tokens = input_tokens + output_tokens

    user_msg, assistant_msg = history_service.add_interaction(
        chat_id=chat.chat_id,
        user_id=user_id,
        case_id=case_id,
        question=question,
        answer=result["answer"],
        sources=result["sources"],
        token_count=total_tokens,  # Ensure this is an integer
        model_used=result.get("model_used"),
        response_time=int(result.get("time_taken", 0) * 1000)  # Convert to ms
    )
    
    print(f"Added user message: {user_msg.message_id}")
    print(f"Added assistant message: {assistant_msg.message_id}")
    
    # 4. Get chat history
    print("\n4. Retrieving chat history...")
    messages, total = chat_manager.get_chat_history(
        chat_id=chat.chat_id,
        user_id=user_id,
        case_id=case_id
    )
    
    print(f"Retrieved {len(messages)} messages out of {total}")
    
    # 5. Add feedback
    print("\n5. Adding feedback...")
    feedback = feedback_manager.add_feedback(
        message_id=assistant_msg.message_id,
        user_id=user_id,
        rating=4,
        comment="Good answer but could be more detailed",
        feedback_type="quality"
    )
    
    print(f"Added feedback: {feedback.id}")
    
    # 6. Send a follow-up query
    print("\n6. Sending follow-up query with summary-only retrieval...")
    follow_up = "What was my previous question"
    
    # Get updated history
    chat_history = chat_manager.get_formatted_history(
        chat_id=chat.chat_id,
        user_id=user_id,
        case_id=case_id
    )
    
    # Process follow-up query with summary-only retrieval
    result = query_engine.query(
        question=follow_up,
        case_id=case_id,
        document_ids=document_ids,
        use_tree=True,  # Try tree-based retrieval
        top_k=3,
        chat_history=chat_history,
        tree_level_filter=[1, 2, 3]  # Use only summary chunks
    )
    
    # Add to history
    input_tokens = int(result.get('input_tokens', 0))
    output_tokens = int(result.get('output_tokens', 0))
    total_tokens = input_tokens + output_tokens

    history_service.add_interaction(
        chat_id=chat.chat_id,
        user_id=user_id,
        case_id=case_id,
        question=follow_up,
        answer=result["answer"],
        sources=result["sources"],
        token_count=total_tokens,  # Ensure this is an integer
        model_used=result.get("model_used"),
        response_time=int(result.get("time_taken", 0) * 1000)
    )
    
    print("\nFollow-up query result:")
    print(f"Answer: {result['answer'][:100]}..." if len(result['answer']) > 100 else result['answer'])
    
    # Count source types
    original_count = sum(1 for s in result['sources'] if s.get('tree_level', 0) == 0)
    summary_count = sum(1 for s in result['sources'] if s.get('tree_level', 0) > 0)
    print(f"Sources: Original Chunks: {original_count}, Summary Chunks: {summary_count}")
    
    # 7. Generate a title based on the conversation
    print("\n7. Generating title based on conversation...")
    title = chat_manager.generate_title(
        chat_id=chat.chat_id,
        user_id=user_id,
        case_id=case_id
    )
    
    print(f"Generated title: {title}")
    
    # 8. Update chat with the new title
    print("\n8. Updating chat with new title...")
    updated_chat = chat_manager.update_chat(
        chat_id=chat.chat_id,
        user_id=user_id,
        case_id=case_id,
        title=title
    )
    
    print(f"Updated chat: {updated_chat.title}")
    
    return {
        "chat_id": chat.chat_id,
        "title": updated_chat.title,
        "message_count": len(messages) + 2  # Including the new interaction
    }

if __name__ == "__main__":
    try:
        # Test the query engine
        query_result = test_query_engine()
        
        # Test complete chat flow
        chat_result = test_chat_flow()
        
        print("\n=== Tests Completed Successfully ===")
        print(f"Query Engine: {query_result['status']}")
        print(f"Chat Flow: Chat {chat_result['chat_id']} with {chat_result['message_count']} messages")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)