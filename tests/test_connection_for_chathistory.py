#!/usr/bin/env python3
"""
Test script to verify database connection for chat repository.
"""

import sys
import os
import logging

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from db.chat_store.repository import ChatRepository
from core.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_db_connection")

def test_repository():
    """Test the ChatRepository class"""
    logger.info("Testing ChatRepository connection and methods")
    
    # Initialize repository
    repo = ChatRepository()
    
    if getattr(repo, "mock_mode", False):
        logger.warning("USING MOCK MODE: Database connection failed, but repository works in mock mode")
    else:
        logger.info("Database connection successful")
    
    # Test create_chat
    try:
        chat = repo.create_chat(
            title="Test Chat",
            user_id="test_user",
            case_id="test_case",
            document_ids=["doc_test1", "doc_test2"],
            settings={"use_tree": True, "top_k": 5}
        )
        
        logger.info(f"Created chat: {chat.chat_id}, Title: {chat.title}")
        
        # Test add_message
        try:
            user_msg = repo.add_message(
                chat_id=chat.chat_id,
                user_id="test_user",
                case_id="test_case",
                role="user",
                content="This is a test message"
            )
            
            logger.info(f"Added user message: {user_msg.message_id}")
            
            assistant_msg = repo.add_message(
                chat_id=chat.chat_id,
                user_id="system",
                case_id="test_case",
                role="assistant",
                content="This is a test response",
                sources=[{"document_id": "doc_test1", "content": "Source content"}],
                token_count=125,
                model_used="test_model"
            )
            
            logger.info(f"Added assistant message: {assistant_msg.message_id}")
            
            # Test get_messages
            try:
                messages, total = repo.get_messages(
                    chat_id=chat.chat_id,
                    limit=10,
                    offset=0
                )
                
                logger.info(f"Retrieved {len(messages)} messages out of {total}")
                
                # Test delete_chat (cleanup)
                try:
                    # deleted = repo.delete_chat(chat.chat_id)
                    logger.info(f"Deleted chat {chat.chat_id}: {deleted}")
                    
                except Exception as e:
                    logger.error(f"Error deleting chat: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error getting messages: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error creating chat: {str(e)}")
    
    logger.info("Repository tests completed")

def test_list_chats():
    """Test listing chats"""
    logger.info("Testing listing chats")
    
    repo = ChatRepository()
    
    try:
        chats, total = repo.list_chats(
            user_id="test_user",
            limit=10,
            offset=0
        )
        
        logger.info(f"Listed {len(chats)} chats out of {total}")
        
        for chat in chats:
            logger.info(f"Chat: {chat.chat_id}, Title: {chat.title}")
        
    except Exception as e:
        logger.error(f"Error listing chats: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting database connection test")
    
    test_repository()
    test_list_chats()
    
    logger.info("Test completed")