# Project Continuation Prompt: Document RAG API Implementation

## Progress Made So Far

We've successfully implemented the first priority from the RAG system's migration plan:

1. Created a **PostgreSQL database schema** with tables for:
   - Chat sessions (chats)
   - Message history (messages)
   - Document associations (chat_documents)
   - User feedback (feedback)
   - User-case access control (user_case_access)

2. Implemented the **Chat Repository** components:
   - `db/chat_store/models.py`: Pydantic models for database entities
   - `db/chat_store/repository.py`: Full repository implementation with CRUD operations
   - `db/chat_store/test_connection.py`: Database connection test script

3. **Verified database connectivity** and operation with our test script

## Next Steps in Implementation Priority

According to the migration plan, the next components to implement are:

1. **Query Engine** (`services/retrieval/query_engine.py`):
   - Implement querying logic against Milvus vector database
   - Add tracking for tokens, response time, etc.
   - Support both flat and hierarchical retrieval methods

2. **Chat History Service** (`services/chat/history.py`):
   - Implement chat history management with message status tracking
   - Create methods for retrieving and formatting chat history
   - Add support for summarization of lengthy histories

3. **Chat Manager** (`services/chat/manager.py`):
   - Create higher-level chat management functionality
   - Implement document loading/unloading for chats
   - Handle chat state transitions

4. **Feedback System** (`services/feedback/manager.py`):
   - Implement feedback recording and retrieval functionality

5. **API Layer** (`api/routes/*.py`):
   - Implement REST endpoints with proper context handling
   - Connect the endpoints to the underlying services

## Instructions For Continuation

Please continue the implementation by:

1. Starting with the Query Engine implementation, as it's the next priority
2. Then moving to the Chat History Service and Chat Manager
3. Following the structure and approach established in our current implementation
4. Maintaining the same emphasis on robust error handling and type safety

Let me know if any specific component should be prioritized or if you'd like to see additional details about any part of the existing code.

- Input and output tokens count differently in `query_engine.py`
- parametrise flat, tree searches, summary and original chunk filters (tree= [1,2] / [0])
- handling of clear history
- Connect title generation to llm
- _is_relevant_to_query method using embedding
- upload job manager and real time stats
- connect case_user_access table

## Advanced

- Retry mechanism and mid-process persistence
case user role documents
