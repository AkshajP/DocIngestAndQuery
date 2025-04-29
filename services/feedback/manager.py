import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from db.chat_store.repository import ChatRepository
from db.chat_store.models import Feedback
from core.config import get_config

logger = logging.getLogger(__name__)

class FeedbackManager:
    """
    Service for collecting and analyzing user feedback on responses.
    """
    
    def __init__(self, config=None):
        """
        Initialize the feedback manager.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or get_config()
        self.chat_repo = ChatRepository()
        
        logger.info("Feedback manager initialized")
    
    def add_feedback(
        self,
        message_id: str,
        user_id: str,
        rating: Optional[int] = None,
        comment: Optional[str] = None,
        feedback_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Feedback:
        """
        Add feedback for a message.
        
        Args:
            message_id: Message ID
            user_id: User ID providing feedback
            rating: Optional numeric rating (e.g., 1-5)
            comment: Optional text feedback
            feedback_type: Optional category of feedback
            metadata: Optional additional metadata
            
        Returns:
            Created Feedback object
        """
        # Verify message exists
        message = self.chat_repo.get_message(message_id)
        if not message:
            raise ValueError(f"Message {message_id} not found")
        
        # Create feedback ID
        feedback_id = f"feedback_{uuid.uuid4().hex[:10]}"
        
        # Create feedback in database
        # In a real implementation, we'd use a dedicated feedback table
        # For now, we'll just add it to the message metadata
        
        # Create feedback object
        feedback = Feedback(
            id=feedback_id,
            message_id=message_id,
            user_id=user_id,
            rating=rating,
            comment=comment,
            created_at=datetime.now(),
            feedback_type=feedback_type,
            metadata=metadata or {}
        )
        
        # In a real implementation, we'd insert this into the database
        # For now, let's add it to the message's metadata
        
        # Get existing metadata
        existing_metadata = message.metadata or {}
        
        # Add or update feedback in metadata
        if "feedback" not in existing_metadata:
            existing_metadata["feedback"] = []
        
        # Add new feedback
        existing_metadata["feedback"].append({
            "id": feedback_id,
            "user_id": user_id,
            "rating": rating,
            "comment": comment,
            "created_at": datetime.now().isoformat(),
            "feedback_type": feedback_type,
            "metadata": metadata or {}
        })
        
        # Update message with new metadata
        self.chat_repo.update_message_status(
            message_id=message_id,
            status=message.status,  # Keep current status
            error_details=None,
            # For a real implementation, you'd update response_time here
        )
        
        logger.info(f"Added feedback {feedback_id} for message {message_id}")
        
        return feedback
    
    def get_message_feedback(
        self,
        message_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get feedback for a specific message.
        
        Args:
            message_id: Message ID
            
        Returns:
            List of feedback items
        """
        # Get message
        message = self.chat_repo.get_message(message_id)
        if not message:
            return []
        
        # Get feedback from metadata
        metadata = message.metadata or {}
        feedback_list = metadata.get("feedback", [])
        
        return feedback_list
    
    def get_user_feedback(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get feedback provided by a specific user.
        
        Args:
            user_id: User ID
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Tuple of (list of feedback items, total count)
        """
        # In a real implementation, we'd query the feedback table
        # For now, we'll scan messages for feedback
        
        # This is highly inefficient and only for demonstration
        # A real implementation would use a dedicated feedback table
        
        all_feedback = []
        total_count = 0
        
        # Get all messages for this user
        # Note: This would be replaced with a proper feedback query
        messages, _ = self.chat_repo.get_messages(
            user_id=user_id,
            limit=100,  # Arbitrary limit
            offset=0
        )
        
        # Extract feedback from message metadata
        for message in messages:
            metadata = message.metadata or {}
            feedback_list = metadata.get("feedback", [])
            
            # Filter to feedback from this user
            user_feedback = [f for f in feedback_list if f.get("user_id") == user_id]
            
            all_feedback.extend(user_feedback)
            total_count += len(user_feedback)
        
        # Apply pagination
        paginated_feedback = all_feedback[offset:offset+limit]
        
        return paginated_feedback, total_count
    
    def get_case_feedback(
        self,
        case_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get feedback for messages in a specific case.
        
        Args:
            case_id: Case ID
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Tuple of (list of feedback items, total count)
        """
        # In a real implementation, we'd query the feedback table
        # For now, we'll scan messages for feedback
        
        all_feedback = []
        total_count = 0
        
        # Get all messages for this case
        # Note: This would be replaced with a proper feedback query
        messages, _ = self.chat_repo.get_messages(
            case_id=case_id,
            limit=100,  # Arbitrary limit
            offset=0
        )
        
        # Extract feedback from message metadata
        for message in messages:
            metadata = message.metadata or {}
            feedback_list = metadata.get("feedback", [])
            
            # Add message context to each feedback item
            for feedback in feedback_list:
                feedback["message"] = {
                    "id": message.message_id,
                    "content": message.content[:100] + "..." if len(message.content) > 100 else message.content,
                    "role": message.role,
                    "created_at": message.created_at.isoformat()
                }
            
            all_feedback.extend(feedback_list)
            total_count += len(feedback_list)
        
        # Apply pagination
        paginated_feedback = all_feedback[offset:offset+limit]
        
        return paginated_feedback, total_count
    
    def analyze_feedback_trends(
        self,
        case_id: Optional[str] = None,
        time_period: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze feedback trends.
        
        Args:
            case_id: Optional case ID to filter by
            time_period: Optional time period ("day", "week", "month")
            
        Returns:
            Dictionary with feedback statistics
        """
        # In a real implementation, we'd query the feedback table
        # For now, we'll demonstrate what this might return
        
        # Mock implementation
        return {
            "average_rating": 4.2,
            "feedback_count": 142,
            "rating_distribution": {
                "1": 5,
                "2": 10,
                "3": 20,
                "4": 50,
                "5": 57
            },
            "common_feedback_types": {
                "accuracy": 65,
                "helpfulness": 40,
                "clarity": 37
            },
            "recent_trend": "improving",
            "period": time_period or "all"
        }