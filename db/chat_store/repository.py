# db/chat_store/repository.py

import psycopg2
import psycopg2.extras
import logging
import uuid
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .models import Chat, Message, Feedback, ChatDocument, UserCaseAccess
from core.config import get_config

logger = logging.getLogger(__name__)

class ChatRepository:
    """Repository for managing chat data in PostgreSQL"""
    
    def __init__(self, db_config=None):
        """
        Initialize chat repository with database configuration.
        
        Args:
            db_config: Optional database configuration override
        """
        self.config = db_config or get_config().database
        self.conn = None
        try:
            self._connect()
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            logger.warning("Operating in mock mode due to database connection failure")
            # We'll use mock data/operations if database connection fails
            self.mock_mode = True
            self.mock_data = {
                "chats": {},
                "messages": {},
                "documents": {}
            }
        else:
            self.mock_mode = False
    
    def _connect(self):
        """Connect to the database"""
        try:
            self.conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                dbname=self.config.dbname,
                user=self.config.user,
                password=self.config.password,
                connect_timeout=self.config.connection_timeout
            )
            logger.info("Connected to chat database")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def _ensure_connection(self):
        """Ensure database connection is active"""
        if self.mock_mode:
            # In mock mode, we don't need a real connection
            return
            
        try:
            if not self.conn or self.conn.closed:
                self._connect()
        except Exception as e:
            logger.error(f"Failed to reconnect to database: {str(e)}")
            logger.warning("Switching to mock mode")
            self.mock_mode = True
            self.mock_data = {
                "chats": {},
                "messages": {},
                "documents": {}
            }
    
    def create_chat(
        self, 
        title: str, 
        user_id: str, 
        case_id: str, 
        document_ids: Optional[List[str]] = None, 
        settings: Optional[Dict[str, Any]] = None, 
        state: str = "open"
    ) -> Chat:
        """
        Create a new chat.
        
        Args:
            title: Chat title
            user_id: User ID creating the chat
            case_id: Case ID the chat belongs to
            document_ids: Optional list of document IDs to load in chat
            settings: Optional chat settings
            state: Initial chat state
            
        Returns:
            Created Chat object
        """
        self._ensure_connection()
        chat_id = f"chat_{uuid.uuid4().hex[:10]}"
        now = datetime.now()
        
        # Handle mock mode
        if self.mock_mode:
            # Create chat in mock data
            chat = Chat(
                chat_id=chat_id,
                title=title,
                user_id=user_id,
                case_id=case_id,
                created_at=now,
                updated_at=now,
                state=state,
                settings=settings or {}
            )
            
            # Store in mock data
            self.mock_data["chats"][chat_id] = {
                "chat_id": chat_id,
                "title": title,
                "user_id": user_id,
                "case_id": case_id,
                "created_at": now,
                "updated_at": now,
                "state": state,
                "settings": settings or {}
            }
            
            # Store document associations if provided
            if document_ids:
                if "chat_documents" not in self.mock_data:
                    self.mock_data["chat_documents"] = {}
                    
                if chat_id not in self.mock_data["chat_documents"]:
                    self.mock_data["chat_documents"][chat_id] = []
                    
                self.mock_data["chat_documents"][chat_id].extend(document_ids)
            
            logger.info(f"Created chat {chat_id} in mock mode")
            return chat
        
        # Normal database mode
        with self.conn.cursor() as cursor:
            # Insert chat
            cursor.execute(
                """
                INSERT INTO chats (chat_id, title, user_id, case_id, created_at, updated_at, state, settings)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    chat_id, 
                    title, 
                    user_id, 
                    case_id, 
                    now, 
                    now, 
                    state, 
                    json.dumps(settings or {})
                )
            )
            chat_row = cursor.fetchone()
            
            # Add documents if provided
            if document_ids:
                for doc_id in document_ids:
                    cursor.execute(
                        """
                        INSERT INTO chat_documents (chat_id, document_id)
                        VALUES (%s, %s)
                        """,
                        (chat_id, doc_id)
                    )
            
            self.conn.commit()
            
            # Create Chat object
            chat = Chat(
                chat_id=chat_id,
                title=title,
                user_id=user_id,
                case_id=case_id,
                created_at=now,
                updated_at=now,
                state=state,
                settings=settings or {}
            )
            
            return chat
    
    def get_chat(self, chat_id: str, user_id: str = None, case_id: str = None) -> Optional[Chat]:
        """
        Get chat by ID with optional user/case filtering.
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID for access control
            case_id: Optional case ID for access control
            
        Returns:
            Chat object or None if not found
        """
        self._ensure_connection()
        
        query = "SELECT * FROM chats WHERE chat_id = %s"
        params = [chat_id]
        
        if user_id:
            query += " AND user_id = %s"
            params.append(user_id)
        
        if case_id:
            query += " AND case_id = %s"
            params.append(case_id)
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Convert to Chat object
            return Chat(
                chat_id=row["chat_id"],
                title=row["title"],
                user_id=row["user_id"],
                case_id=row["case_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                state=row["state"],
                settings=row["settings"]
            )
    
    def update_chat(
        self, 
        chat_id: str, 
        user_id: str = None, 
        case_id: str = None, 
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
        self._ensure_connection()
        
        # First check if chat exists with access control
        chat = self.get_chat(chat_id, user_id, case_id)
        if not chat:
            return None
        
        # Build update query
        update_parts = []
        params = []
        
        if title is not None:
            update_parts.append("title = %s")
            params.append(title)
        
        if settings is not None:
            update_parts.append("settings = %s")
            params.append(json.dumps(settings))
        
        if state is not None:
            update_parts.append("state = %s")
            params.append(state)
        
        # Always update updated_at
        update_parts.append("updated_at = %s")
        now = datetime.now()
        params.append(now)
        
        # Add chat_id as last parameter
        params.append(chat_id)
        
        if not update_parts:
            return chat  # Nothing to update
        
        # Execute update
        with self.conn.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE chats
                SET {", ".join(update_parts)}
                WHERE chat_id = %s
                RETURNING *
                """,
                params
            )
            
            self.conn.commit()
            
            # Update the chat object
            if title is not None:
                chat.title = title
            if settings is not None:
                chat.settings = settings
            if state is not None:
                chat.state = state
            chat.updated_at = now
            
            return chat
    
    def delete_chat(self, chat_id: str, user_id: str = None, case_id: str = None) -> bool:
        """
        Delete a chat.
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID for access control
            case_id: Optional case ID for access control
            
        Returns:
            True if deleted, False if not found
        """
        self._ensure_connection()
        
        query = "DELETE FROM chats WHERE chat_id = %s"
        params = [chat_id]
        
        if user_id:
            query += " AND user_id = %s"
            params.append(user_id)
        
        if case_id:
            query += " AND case_id = %s"
            params.append(case_id)
        
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            rows_deleted = cursor.rowcount
            self.conn.commit()
            
            return rows_deleted > 0
    
    def list_chats(
        self, 
        user_id: str = None, 
        case_id: str = None, 
        state: str = None, 
        limit: int = 50, 
        offset: int = 0
    ) -> Tuple[List[Chat], int]:
        """
        List chats with optional filtering.
        
        Args:
            user_id: Optional user ID filter
            case_id: Optional case ID filter
            state: Optional state filter
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Tuple of (list of Chat objects, total count)
        """
        self._ensure_connection()
        
        # Build query conditions
        conditions = []
        params = []
        
        if user_id:
            conditions.append("user_id = %s")
            params.append(user_id)
        
        if case_id:
            conditions.append("case_id = %s")
            params.append(case_id)
        
        if state:
            conditions.append("state = %s")
            params.append(state)
        
        # Build where clause
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        
        # Get total count
        count_query = f"SELECT COUNT(*) FROM chats {where_clause}"
        
        # Build main query
        query = f"""
            SELECT * FROM chats
            {where_clause}
            ORDER BY updated_at DESC
            LIMIT %s OFFSET %s
        """
        
        # Add pagination params
        params.extend([limit, offset])
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            # Get total count
            cursor.execute(count_query, params[:-2] if params else [])
            total_count = cursor.fetchone()[0]
            
            # Get chats
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to Chat objects
            chats = [
                Chat(
                    chat_id=row["chat_id"],
                    title=row["title"],
                    user_id=row["user_id"],
                    case_id=row["case_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    state=row["state"],
                    settings=row["settings"]
                )
                for row in rows
            ]
            
            return chats, total_count
    
    def add_message(
        self, 
        chat_id: str, 
        user_id: str, 
        case_id: str, 
        role: str, 
        content: str, 
        sources: Optional[List[Dict[str, Any]]] = None, 
        status: str = "sent", 
        token_count: Optional[int] = None, 
        model_used: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to a chat.
        
        Args:
            chat_id: Chat ID
            user_id: User ID
            case_id: Case ID
            role: Message role (user, assistant, system)
            content: Message content
            sources: Optional sources for assistant messages
            status: Message status
            token_count: Optional token count
            model_used: Optional model used
            metadata: Optional metadata
            
        Returns:
            Created Message object
        """
        self._ensure_connection()
        message_id = f"msg_{uuid.uuid4().hex[:10]}"
        now = datetime.now()
        
        # Update chat's updated_at timestamp
        with self.conn.cursor() as cursor:
            cursor.execute(
                "UPDATE chats SET updated_at = %s WHERE chat_id = %s",
                (now, chat_id)
            )
            
            # Insert message
            cursor.execute(
                """
                INSERT INTO messages (
                    message_id, chat_id, user_id, case_id, role, content, created_at,
                    sources, metadata, status, token_count, model_used
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    message_id, chat_id, user_id, case_id, role, content, now,
                    json.dumps(sources) if sources else None,
                    json.dumps(metadata or {}),
                    status, token_count, model_used
                )
            )
            
            self.conn.commit()
            
            # Create Message object
            message = Message(
                message_id=message_id,
                chat_id=chat_id,
                user_id=user_id,
                case_id=case_id,
                role=role,
                content=content,
                created_at=now,
                sources=sources,
                metadata=metadata or {},
                status=status,
                token_count=token_count,
                model_used=model_used
            )
            
            return message
    
    def update_message_status(
        self, 
        message_id: str, 
        status: str, 
        error_details: Optional[Dict[str, Any]] = None, 
        response_time: Optional[int] = None,
        sources: Optional[List[Dict[str, Any]]] = None,  # Add this
        token_count: Optional[int] = None,  # Add this
        model_used: Optional[str] = None  # Add this
    ) -> Optional[Message]:
        """
        Update message status and related fields.
        
        Args:
            message_id: Message ID
            status: New status
            error_details: Optional error details for failed status
            response_time: Optional response time
            sources: Optional sources used for the response
            token_count: Optional token count used
            model_used: Optional model name used
            
        Returns:
            Updated Message object or None if not found
        """
        self._ensure_connection()
        
        update_parts = ["status = %s"]
        params = [status]
        
        if error_details is not None:
            update_parts.append("error_details = %s")
            params.append(json.dumps(error_details))
        
        if response_time is not None:
            update_parts.append("response_time = %s")
            params.append(response_time)
            
        if sources is not None:
            update_parts.append("sources = %s")
            params.append(json.dumps(sources))
            
        if token_count is not None:
            update_parts.append("token_count = %s")
            params.append(token_count)
            
        if model_used is not None:
            update_parts.append("model_used = %s")
            params.append(model_used)
        
        # Add message_id as last parameter
        params.append(message_id)
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(
                f"""
                UPDATE messages
                SET {", ".join(update_parts)}
                WHERE message_id = %s
                RETURNING *
                """,
                params
            )
            
            row = cursor.fetchone()
            self.conn.commit()
            
            if not row:
                return None
            
            # Convert to Message object
            return Message(
                message_id=row["message_id"],
                chat_id=row["chat_id"],
                user_id=row["user_id"],
                case_id=row["case_id"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
                sources=row["sources"],
                metadata=row["metadata"],
                status=row["status"],
                token_count=row["token_count"],
                model_used=row["model_used"],
                error_details=row["error_details"],
                response_time=row["response_time"]
            )
    
    def get_messages(
        self, 
        chat_id: str, 
        user_id: str = None, 
        case_id: str = None, 
        limit: int = 50, 
        offset: int = 0
    ) -> Tuple[List[Message], int]:
        """
        Get messages for a chat.
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID for access control
            case_id: Optional case ID for access control
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Tuple of (list of Message objects, total count)
        """
        self._ensure_connection()
        
        # Build query conditions
        conditions = ["chat_id = %s"]
        params = [chat_id]
        
        if user_id:
            conditions.append("user_id = %s")
            params.append(user_id)
        
        if case_id:
            conditions.append("case_id = %s")
            params.append(case_id)
        
        # Build where clause
        where_clause = "WHERE " + " AND ".join(conditions)
        
        # Get total count
        count_query = f"SELECT COUNT(*) FROM messages {where_clause}"
        
        # Build main query
        query = f"""
            SELECT * FROM messages
            {where_clause}
            ORDER BY created_at ASC
            LIMIT %s OFFSET %s
        """
        
        # Add pagination params
        params.extend([limit, offset])
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            # Get total count
            cursor.execute(count_query, params[:-2])
            total_count = cursor.fetchone()[0]
            
            # Get messages
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to Message objects
            messages = [
                Message(
                    message_id=row["message_id"],
                    chat_id=row["chat_id"],
                    user_id=row["user_id"],
                    case_id=row["case_id"],
                    role=row["role"],
                    content=row["content"],
                    created_at=row["created_at"],
                    sources=row["sources"],
                    metadata=row["metadata"],
                    status=row["status"],
                    token_count=row["token_count"],
                    model_used=row["model_used"],
                    error_details=row["error_details"],
                    response_time=row["response_time"]
                )
                for row in rows
            ]
            
            return messages, total_count
    
    def get_message(self, message_id: str) -> Optional[Message]:
        """
        Get a message by ID.
        
        Args:
            message_id: Message ID
            
        Returns:
            Message object or None if not found
        """
        self._ensure_connection()
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM messages WHERE message_id = %s",
                (message_id,)
            )
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Convert to Message object
            return Message(
                message_id=row["message_id"],
                chat_id=row["chat_id"],
                user_id=row["user_id"],
                case_id=row["case_id"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
                sources=row["sources"],
                metadata=row["metadata"],
                status=row["status"],
                token_count=row["token_count"],
                model_used=row["model_used"],
                error_details=row["error_details"],
                response_time=row["response_time"]
            )
    
    def update_chat_documents(
        self, 
        chat_id: str, 
        user_id: str = None, 
        case_id: str = None, 
        add_docs: Optional[List[str]] = None, 
        remove_docs: Optional[List[str]] = None
    ) -> bool:
        """
        Update documents associated with a chat.
        
        Args:
            chat_id: Chat ID
            user_id: Optional user ID for access control
            case_id: Optional case ID for access control
            add_docs: List of document IDs to add
            remove_docs: List of document IDs to remove
            
        Returns:
            True if successful, False if chat not found
        """
        self._ensure_connection()
        
        # Check if chat exists with access control
        chat = self.get_chat(chat_id, user_id, case_id)
        if not chat:
            return False
        
        with self.conn.cursor() as cursor:
            # Remove documents if specified
            if remove_docs:
                placeholders = ', '.join(['%s'] * len(remove_docs))
                cursor.execute(
                    f"""
                    DELETE FROM chat_documents
                    WHERE chat_id = %s AND document_id IN ({placeholders})
                    """,
                    [chat_id] + remove_docs
                )
            
            # Add documents if specified
            if add_docs:
                for doc_id in add_docs:
                    try:
                        cursor.execute(
                            """
                            INSERT INTO chat_documents (chat_id, document_id)
                            VALUES (%s, %s)
                            ON CONFLICT (chat_id, document_id) DO NOTHING
                            """,
                            (chat_id, doc_id)
                        )
                    except Exception as e:
                        logger.error(f"Error adding document {doc_id} to chat {chat_id}: {str(e)}")
            
            # Update chat's updated_at timestamp
            cursor.execute(
                "UPDATE chats SET updated_at = %s WHERE chat_id = %s",
                (datetime.now(), chat_id)
            )
            
            self.conn.commit()
            
            return True
        
    def update_message_content(
        self, 
        message_id: str, 
        content: str
    ) -> Optional[Message]:
        """
        Update just the content of a message, useful for streaming updates.
        
        Args:
            message_id: Message ID
            content: New message content
            
        Returns:
            Updated Message object or None if not found
        """
        self._ensure_connection()
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(
                """
                UPDATE messages
                SET content = %s
                WHERE message_id = %s
                RETURNING *
                """,
                (content, message_id)
            )
            
            row = cursor.fetchone()
            self.conn.commit()
            
            if not row:
                return None
            
            # Convert to Message object
            return Message(
                message_id=row["message_id"],
                chat_id=row["chat_id"],
                user_id=row["user_id"],
                case_id=row["case_id"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
                sources=row["sources"],
                metadata=row["metadata"],
                status=row["status"],
                token_count=row["token_count"],
                model_used=row["model_used"],
                error_details=row["error_details"],
                response_time=row["response_time"]
            )