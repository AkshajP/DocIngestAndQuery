import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from db.chat_store.repository import ChatRepository
from db.chat_store.models import Message, MessageRole, MessageStatus
from services.retrieval.context_models import ChatHistory, HistoryItem
from core.config import get_config

logger = logging.getLogger(__name__)

class ChatHistoryService:
    """
    Service for managing chat history with support for summarization
    and formatting for different contexts.
    """
    
    def __init__(self, config=None, chat_repo=None):
        """
        Initialize the chat history service.
        
        Args:
            config: Optional configuration override
            chat_repo: Optional injected ChatRepository instance
        """
        self.config = config or get_config()
        
        # Use injected repository or create new one (for backward compatibility)
        self.chat_repo = chat_repo or ChatRepository()
        
        # Initialize LLM for summarization
        self.llm = OllamaLLM(
            model=self.config.ollama.model,
            base_url=self.config.ollama.base_url,
            temperature=0.1
        )
        
        # Summarization prompt
        self.summary_prompt = PromptTemplate(
            template="""
            Summarize the following conversation history. Focus on key information, questions, and findings.
            Preserve details that might be relevant to future questions.
            Pay special attention to information marked as high importance.
            
            CONVERSATION HISTORY:
            {history}
            
            SUMMARY:
            """,
            input_variables=["history"]
        )
        
        logger.info("Chat history service initialized")
    
    def add_interaction(
        self,
        chat_id: str,
        user_id: str,
        case_id: str,
        question: str,
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        token_count: Optional[int] = None,
        model_used: Optional[str] = None,
        response_time: Optional[int] = None
    ) -> Tuple[Message, Message]:
        """
        Add a Q&A interaction to chat history.
        
        Args:
            chat_id: Chat ID
            user_id: User ID
            case_id: Case ID
            question: User question
            answer: System answer
            sources: Optional sources used for the answer
            token_count: Optional token count
            model_used: Optional model used
            response_time: Optional response time in milliseconds
            
        Returns:
            Tuple of (user message, assistant message)
        """
        # Add user message
        user_message = self.chat_repo.add_message(
            chat_id=chat_id,
            user_id=user_id,
            case_id=case_id,
            role=MessageRole.USER,
            content=question,
            status=MessageStatus.COMPLETED
        )
        
        # Add assistant message with sources
        assistant_message = self.chat_repo.add_message(
            chat_id=chat_id,
            user_id=user_id,
            case_id=case_id,
            role=MessageRole.ASSISTANT,
            content=answer,
            sources=sources,
            status=MessageStatus.COMPLETED,
            token_count=token_count,
            model_used=model_used,
            metadata={
                "response_time": response_time
            }
        )
        
        logger.info(f"Added Q&A interaction to chat {chat_id}")
        
        return user_message, assistant_message
    
    def get_history(
        self,
        chat_id: str,
        user_id: Optional[str] = None,
        case_id: Optional[str] = None,
        limit: int = 20,
        include_metadata: bool = False
    ) -> ChatHistory:
        """
        Get chat history formatted for use in prompts or display.
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID for access control
            case_id: Optional case ID for access control
            limit: Maximum number of messages to retrieve
            include_metadata: Whether to include message metadata
            
        Returns:
            ChatHistory object
        """
        # Get messages from repository
        messages, total = self.chat_repo.get_messages(
            chat_id=chat_id,
            user_id=user_id,
            case_id=case_id,
            limit=limit,
            offset=0
        )
        
        # Format as ChatHistory object
        history_items = []
        
        for message in messages:
            metadata = {}
            
            if include_metadata:
                metadata = {
                    "status": message.status,
                    "created_at": message.created_at.isoformat(),
                    "message_id": message.message_id
                }
                
                if message.sources:
                    metadata["has_sources"] = True
                
                if message.token_count:
                    metadata["token_count"] = message.token_count
                
                if message.model_used:
                    metadata["model_used"] = message.model_used
            
            history_items.append(HistoryItem(
                role=message.role,
                content=message.content,
                metadata=metadata
            ))
        
        # Create summary if we hit the limit and there are more messages
        summary = None
        if total > limit:
            summary = f"This chat has {total} messages in total. Showing the most recent {limit}."
        
        return ChatHistory(
            items=history_items,
            summary=summary
        )
    
    def summarize_history(
        self,
        chat_id: str,
        user_id: Optional[str] = None,
        case_id: Optional[str] = None,
        query: Optional[str] = None
    ) -> str:
        """
        Generate a summary of chat history.
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID for access control
            case_id: Optional case ID for access control
            query: Optional current query for context-aware summarization
            
        Returns:
            Summary text
        """
        # Get full history
        messages, total = self.chat_repo.get_messages(
            chat_id=chat_id,
            user_id=user_id,
            case_id=case_id,
            limit=20,  # Get more messages for summarization
            offset=0
        )
        
        if not messages:
            return ""
        
        # Format history for summarization
        history_text = ""
        for message in messages:
            role_name = "USER" if message.role == MessageRole.USER else "ASSISTANT"
            
            # Create a simplified interaction dict based on message role
            if message.role == MessageRole.USER:
                interaction = {"question": message.content, "answer": ""}
            else:
                interaction = {"question": "", "answer": message.content}
                
            # Determine importance using updated function
            importance = "high" if query and self.is_relevant_to_query(
                interaction=interaction,
                query=query
            ) else "normal"
            
            history_text += f"{role_name}: {message.content}\n"
            history_text += f"IMPORTANCE: {importance}\n\n"
        
        # Generate summary
        try:
            summary_chain = self.summary_prompt | self.llm | StrOutputParser()
            summary = summary_chain.invoke({"history": history_text})
            
            logger.info(f"Generated summary for chat {chat_id} with {len(messages)} messages")
            
            return summary
        except Exception as e:
            logger.error(f"Error generating history summary: {str(e)}")
            # Fallback to simple summary
            return f"Previous chat with {total} messages about {self._extract_topics(history_text)}"
    
    def clear_history(
        self,
        chat_id: str,
        user_id: Optional[str] = None,
        case_id: Optional[str] = None
    ) -> bool:
        """
        Clear chat history (archive messages).
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID for access control
            case_id: Optional case ID for access control
            
        Returns:
            True if successful
        """
        # In a real implementation, we'd archive messages
        # For now, let's just add a system message indicating history was cleared
        try:
            # Add system message
            self.chat_repo.add_message(
                chat_id=chat_id,
                user_id=user_id or "system",
                case_id=case_id or "system",
                role=MessageRole.SYSTEM,
                content="Chat history was cleared at this point.",
                status=MessageStatus.COMPLETED
            )
            
            logger.info(f"Cleared history for chat {chat_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error clearing chat history: {str(e)}")
            return False
    
    def update_message_status(
        self,
        message_id: str,
        status: MessageStatus,
        error_details: Optional[Dict[str, Any]] = None
    ) -> Optional[Message]:
        """
        Update the status of a message.
        
        Args:
            message_id: Message ID
            status: New status
            error_details: Optional error details for failed status
            
        Returns:
            Updated Message or None if not found
        """
        return self.chat_repo.update_message_status(
            message_id=message_id,
            status=status,
            error_details=error_details
        )
    
    def is_relevant_to_query(self, interaction: Dict[str, str], query: str, threshold: float = 0.3) -> bool:
        """
        Determine if a past interaction is relevant to the current query using embedding similarity.
        
        Args:
            interaction: Dictionary containing "question" and "answer" keys
            query: Current user query
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            True if the interaction is relevant to the query, False otherwise
        """
        if not self.embedding_service:
            # Lazy-initialize embedding service if not already done
            from services.ml.embeddings import EmbeddingService
            self.embedding_service = EmbeddingService(
                model_name=self.config.ollama.embed_model,
                base_url=self.config.ollama.base_url
            )
        
        try:
            # Create a combined representation of the past interaction
            past_text = f"{interaction.get('question', '')} {interaction.get('answer', '')}"
            
            # Generate embeddings for past interaction and current query
            past_embedding = self.embedding_service.generate_embedding(past_text)
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Calculate cosine similarity
            import numpy as np
            dot_product = np.dot(past_embedding, query_embedding)
            norm1 = np.linalg.norm(past_embedding)
            norm2 = np.linalg.norm(query_embedding)
            
            if norm1 == 0 or norm2 == 0:
                return False
                
            similarity = dot_product / (norm1 * norm2)
            
            # Log similarity for debugging if needed
            logger.debug(f"Similarity between query and past interaction: {similarity:.4f}")
            
            # Return True if similarity is above threshold
            return similarity > threshold
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {str(e)}")
            # Default to including the interaction if there's an error
            return True
    
    def _extract_topics(self, text: str) -> str:
        """
        Extract main topics from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Comma-separated topics
        """
        # Simple keyword extraction for now
        words = text.lower().split()
        # Remove common words
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with"}
        filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Get most common words
        from collections import Counter
        common_words = Counter(filtered_words).most_common(5)
        
        if not common_words:
            return "various topics"
            
        topics = [word for word, _ in common_words]
        return ", ".join(topics)