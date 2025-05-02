# tests/test_chat_routes.py

import sys
import os
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import router and dependencies
from api.routes.chat_routes import router
from fastapi import FastAPI
from api.models.chat_models import ChatCreateRequest, ChatUpdateRequest, ChatDocumentsUpdateRequest
from api.models.query_models import QueryRequest, RegenerateResponseRequest
from db.document_store.repository import DocumentMetadataRepository
from db.chat_store.models import Message, MessageRole

# Create a test app with the router
app = FastAPI()
app.include_router(router)
client = TestClient(app)

# Mock authentication dependencies
def override_get_current_user():
    return "test_user"

def override_get_current_case():
    return "test_case"

app.dependency_overrides = {
    "api.routes.chat_routes.get_current_user": override_get_current_user,
    "api.routes.chat_routes.get_current_case": override_get_current_case
}

# Test fixtures
@pytest.fixture
def mock_chat_manager():
    """Mock the ChatManager class"""
    with patch("api.routes.chat_routes.ChatManager") as mock:
        # Set up return values for the mock methods
        manager_instance = mock.return_value
        
        # create_chat mock
        chat_mock = MagicMock()
        chat_mock.chat_id = "chat_12345"
        chat_mock.title = "Test Chat"
        chat_mock.created_at = "2025-01-01T12:00:00"
        chat_mock.updated_at = "2025-01-01T12:00:00"
        manager_instance.create_chat.return_value = chat_mock
        
        # get_chat mock
        manager_instance.get_chat.return_value = chat_mock
        
        # list_chats mock
        manager_instance.list_chats.return_value = ([chat_mock], 1)
        
        # update_chat mock
        manager_instance.update_chat.return_value = chat_mock
        
        # delete_chat mock
        manager_instance.delete_chat.return_value = True
        
        # get_chat_history mock
        message_mock = {
            "id": "msg_12345",
            "role": "user",
            "content": "Test message",
            "created_at": "2025-01-01T12:00:00"
        }
        manager_instance.get_chat_history.return_value = ([
            {
                "id": "msg_user_12345",
                "role": "user",
                "content": "Test question",
                "created_at": "2025-01-01T12:00:00"
            },
            {
                "id": "msg_12345",  # This ID must match the message_id in the test
                "role": "assistant",
                "content": "Original answer",
                "created_at": "2025-01-01T12:01:00"
            }
        ], 2)
        
        # get_formatted_history mock
        manager_instance.get_formatted_history.return_value = "Previous conversation: Human: Test message"
        
        # get_chat_documents mock
        manager_instance.get_chat_documents.return_value = ["doc_12345"]
        
        # update_chat_documents mock
        manager_instance.update_chat_documents.return_value = True
        
        # generate_title mock
        manager_instance.generate_title.return_value = "Generated Title"
        
        yield mock

@pytest.fixture
def mock_query_engine():
    """Mock the QueryEngine class"""
    with patch("api.routes.chat_routes.QueryEngine") as mock:
        # Set up return values for the mock methods
        engine_instance = mock.return_value
        
        # query mock
        engine_instance.query.return_value = {
            "status": "success",
            "answer": "This is a test answer",
            "sources": [
                {
                    "document_id": "doc_12345",
                    "content": "Source content",
                    "score": 0.9
                }
            ],
            "time_taken": 1.5,
            "input_tokens": 100,
            "output_tokens": 50,
            "model_used": "test_model"
        }
        
        yield mock

@pytest.fixture
def mock_history_service():
    """Mock the ChatHistoryService class"""
    with patch("api.routes.chat_routes.ChatHistoryService") as mock:
        history_instance = mock.return_value
        
        # Mock the message repository and the message
        history_instance.chat_repo = MagicMock()
        
        # add_message mock with a string message_id
        message_mock = MagicMock()
        message_mock.message_id = "msg_12345"
        history_instance.chat_repo.add_message.return_value = message_mock
        
        # get_message mock
        history_instance.chat_repo.get_message.return_value = Message(
            message_id="msg_12345",
            chat_id="chat_12345",
            user_id="test_user",
            case_id="test_case",
            role=MessageRole.ASSISTANT,
            content="Test answer",
            status="completed"
        )
        
        # add_interaction mock - return actual strings for message IDs
        history_instance.add_interaction.return_value = (
            Message(
                message_id="msg_user_12345",
                chat_id="chat_12345",
                user_id="test_user",
                case_id="test_case",
                role=MessageRole.USER,
                content="Test question",
                status="completed"
            ),
            Message(
                message_id="msg_assistant_12345",
                chat_id="chat_12345",
                user_id="test_user",
                case_id="test_case",
                role=MessageRole.ASSISTANT,
                content="Test answer",
                status="completed"
            )
        )
        
        yield mock

@pytest.fixture
def mock_document_repository():
    """Mock the DocumentMetadataRepository class"""
    with patch("api.routes.chat_routes.DocumentMetadataRepository") as mock:
        # Set up return values for the mock methods
        repo_instance = mock.return_value
        
        # get_document mock
        repo_instance.get_document.return_value = {
            "document_id": "doc_12345",
            "original_filename": "test_document.pdf",
            "status": "processed"
        }
        
        yield mock

# Tests for each endpoint
@pytest.mark.usefixtures("mock_chat_manager", "mock_history_service", "mock_document_repository")
def test_create_chat():
    """Test create_chat endpoint"""
    request_data = {
        "title": "Test Chat",
        "loaded_documents": [
            {
                "document_id": "doc_12345",
                "title": "Test Document"
            }
        ]
    }
    
    response = client.post("/ai/chats", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "id" in data
    assert data["title"] == "Test Chat"
    assert "loaded_documents" in data
    assert "history" in data

@pytest.mark.usefixtures("mock_chat_manager")
def test_list_chats():
    """Test list_chats endpoint"""
    response = client.get("/ai/chats")
    assert response.status_code == 200
    
    data = response.json()
    assert "chats" in data
    assert "pagination" in data
    assert len(data["chats"]) > 0

@pytest.mark.usefixtures("mock_chat_manager", "mock_document_repository")
def test_get_chat():
    """Test get_chat endpoint"""
    response = client.get("/ai/chats/chat_12345")
    assert response.status_code == 200
    
    data = response.json()
    assert data["id"] == "chat_12345"
    assert "title" in data
    assert "loaded_documents" in data
    assert "history" in data

@pytest.mark.usefixtures("mock_chat_manager")
def test_update_chat():
    """Test update_chat endpoint"""
    request_data = {
        "title": "Updated Chat Title"
    }
    
    response = client.patch("/ai/chats/chat_12345", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["id"] == "chat_12345"
    assert "title" in data

@pytest.mark.usefixtures("mock_chat_manager")
def test_delete_chat():
    """Test delete_chat endpoint"""
    response = client.delete("/ai/chats/chat_12345")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert "message" in data

@pytest.mark.usefixtures("mock_chat_manager")
def test_get_chat_history():
    """Test get_chat_history endpoint"""
    response = client.get("/ai/chats/chat_12345/history")
    assert response.status_code == 200
    
    data = response.json()
    assert "messages" in data
    assert "pagination" in data
    assert len(data["messages"]) > 0

@pytest.mark.usefixtures("mock_chat_manager", "mock_document_repository")
def test_update_chat_documents():
    """Test update_chat_documents endpoint"""
    request_data = {
        "add": ["doc_12345"],
        "remove": ["doc_67890"]
    }
    
    response = client.put("/ai/chats/chat_12345/documents", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert "loaded_documents" in data

@pytest.mark.usefixtures("mock_chat_manager", "mock_query_engine", "mock_history_service", "mock_document_repository")
def test_submit_query():
    """Test submit_query endpoint"""
    request_data = {
        "question": "Test question",
        "use_tree": False,
        "top_k": 5
    }
    
    # Patch the QueryResponse for better control
    with patch("api.routes.chat_routes.QueryResponse") as mock_response:
        # Make sure the mock returns a proper Pydantic model
        mock_response.return_value = {
            "id": "msg_12345",
            "answer": "This is a test answer",
            "sources": [
                {
                    "document_id": "doc_12345",
                    "content": "Source content",
                    "score": 0.9,
                    "metadata": {}
                }
            ],
            "processing_stats": {
                "time_taken": 1.5,
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "retrieval_time": 0.5,
                "llm_time": 1.0,
                "method": "flat",
                "model_used": "test_model"
            }
        }
        
        response = client.post("/ai/chats/chat_12345/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert "answer" in data
        assert "sources" in data
        assert "processing_stats" in data

@pytest.mark.usefixtures("mock_chat_manager")
def test_generate_chat_title():
    """Test generate_chat_title endpoint"""
    response = client.get("/ai/chats/chat_12345/title")
    assert response.status_code == 200
    
    data = response.json()
    assert "title" in data

@pytest.mark.usefixtures("mock_chat_manager", "mock_query_engine", "mock_history_service")
def test_regenerate_response():
    """Test regenerate_response endpoint"""
    request_data = {
        "use_tree": True,
        "top_k": 3
    }
    
    # Patch the required objects for more controlled testing
    with patch("api.routes.chat_routes.QueryResponse") as mock_response:
        # Make sure the mock returns a proper Pydantic model
        mock_response.return_value = {
            "id": "msg_12345",
            "answer": "This is a regenerated answer",
            "sources": [
                {
                    "document_id": "doc_12345",
                    "content": "Source content",
                    "score": 0.95,
                    "metadata": {}
                }
            ],
            "processing_stats": {
                "time_taken": 1.2,
                "input_tokens": 80,
                "output_tokens": 40,
                "total_tokens": 120,
                "retrieval_time": 0.4,
                "llm_time": 0.8,
                "method": "tree",
                "model_used": "test_model"
            }
        }
        
        response = client.post("/ai/chats/chat_12345/messages/msg_12345/regenerate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert "answer" in data
        assert "sources" in data
        assert "processing_stats" in data

if __name__ == "__main__":
    pytest.main(["-v"])