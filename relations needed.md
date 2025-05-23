# Todo

- [x] Get idea for code structure and architecture for refractoring to new modular enterprise grade structure
- [x] Create Pydantic Models for passing contexts (UserContext, LLMResponse, Prompt, etc)
- [x] Try getting all configs from one json file (embedding model, dimension, LLMmodel, chunk size, dbname)
- [x] Move to schema with additional explicit fields for vector database for easy filtering (including chunk_text, chunk_type (real, synthetic), page_number, chunk_type (text, table, image), chunk)
- [x] Refractor to save embeddings to vector store with partition (collection > partition > entity(embedding))
- [ ] Hardcode outputs for some dependent steps to stub to move forward with project
- [ ] Discuss tool choices (celery +redis/rabbitmq, etc)

---

docs it parallely
logs research  - logging for what?
getting query part up on API architecture
schema and data model to use
context to get from dash
api access parameters

---

Process
RAG->intf->add history feat->modify retriever (for docs)->modify retriever (graph)

---

```s
document_rag_api/
├── api/                                   # API Layer
│   ├── routes/                            # API Routes
│   │   ├── document_routes.py             # Document management endpoints
│   │   ├── chat_routes.py                 # Chat & query endpoints
│   │   └── admin_routes.py                # Admin endpoints (placeholder)
│   ├── models/                            # Pydantic Models
│   │   ├── document_models.py             # Document request/response models
│   │   ├── chat_models.py                 # Chat request/response models
│   │   ├── query_models.py                # Query models
│   │   └── admin_models.py                # Admin models
│   └── middleware/                        # API Middleware
│       ├── auth.py                        # Authentication middleware
│       └── logging.py                     # Request logging middleware
├── core/                                  # Core Application
│   ├── config.py                          # Centralized configuration
│   ├── logging.py                         # Logging setup
│   └── exceptions.py                      # Custom exceptions
├── services/                              # Business Logic
│   ├── document/                          # Document Processing
│   │   ├── processor.py                   # Main document processor
│   │   ├── chunker.py                     # Document chunking
│   │   └── storage.py                     # Storage adapters (S3, local)
│   ├── chat/                              # Chat Management
│   │   ├── manager.py                     # Chat session management
│   │   └── history.py                     # Chat history handling
│   ├── retrieval/                         # Retrieval System
│   │   ├── raptor.py                      # RAPTOR implementation
│   │   ├── query_engine.py                # Query processing
│   │   └── semantics.py                   # Semantic search
│   ├── pdf/                               # PDF Services
│   │   ├── highlighter.py                 # PDF highlighting
│   │   └── extractor.py                   # MinerU ingestion
│   └── ml/                                # ML Services
│       ├── embeddings.py                  # Embedding generation
│       └── model_manager.py               # Model management
├── db/                                    # Database Layer
│   ├── vector_store/                      # Vector Database
│   │   ├── milvus_client.py               # Milvus implementation
│   │   └── schemas.py                     # Vector DB schemas with partitions
│   ├── document_store/                    # Document Metadata Storage
│   │   └── repository.py                  # Document metadata
│   └── chat_store/                        # Chat Storage
│       └── repository.py                  # Chat/message storage
├── tasks/                                 # Simple Task Queue
│   ├── queue.py                           # Thread-based task queue
│   ├── document_jobs.py                   # Document processing jobs
│   └── embedding_jobs.py                  # Embedding generation jobs
├── utils/                                 # Utilities
│   ├── file_utils.py                      # File handling
│   └── text_utils.py                      # Text processing
├── tests/                                 # Testing
├── main.py                                # Application entry point
├── Dockerfile
├── docker-compose.yml
└── config.json                            # Centralized config
```

User role 


```json
{
    "case_id": {
      "docs": [
        {
          "doc_src_path": "",
          "doc_id": "",
          "doc_desc": "",
          "doc_date": "",
          "reference": ""
        }
      ]
    }
  }
```

case title- Claimant vs Respondent || will get this info soon

% TEMP

Priority 1: Parametrize Search Options in Query Interface
This is a high-impact, medium-complexity task that will significantly improve the user experience by allowing UI toggles for search preferences.

Add parameters to control flat vs. tree traversal in query_engine.py
Implement filters for tree_level (0 for original chunks, >0 for summaries)
Update API endpoints to accept these parameters
Connect to UI toggles

Priority 2: Implement Token Counting
Critical for monitoring and potentially billing, with relatively low implementation complexity:

Track both input tokens (prompt) and output tokens (response)
Add tokenizers appropriate for the models being used
Include token counts in response metadata

Priority 3: Implement Relevance Method
This will improve retrieval quality by ensuring returned chunks are relevant:

Add _is_relevant_to_query using embedding similarity
Implement proper threshold for relevance determination
Apply this filter during retrieval to improve results

Priority 4: Stream LLM Responses
Enhances user experience by showing responses as they're generated:

Modify LLM wrappers to support streaming
Update API endpoints to handle streaming responses
Implement client-side handling of chunked data

Priority 5: Connect Title Generation
Good user experience enhancement:

Create prompt for generating chat titles based on initial messages
Connect to chat history and LLM
Add API endpoint for title generation

Priority 6: Upload Job Manager and Real-time Stats
Important for monitoring document processing:

Implement job tracking endpoints
Create real-time statistics reporting
Develop progress tracking and status updates

Priority 7: Case-User Access Implementation
Security feature that can be implemented after core functionality:

Connect case_user_access table to authorization system
Implement permission checks in API endpoints
Create middleware for access control


the actual API
streaming response
Job manager
case user
