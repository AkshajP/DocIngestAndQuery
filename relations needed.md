# Todo

- [x] Get idea for code structure and architecture for refractoring to new modular enterprise grade structure
- [ ] Create Pydantic Models for passing contexts (UserContext, LLMResponse, Prompt, etc)
- [ ] Try getting all configs from one json file (embedding model, dimension, LLMmodel, chunk size, dbname)
- [ ] Move to schema with additional explicit fields for vector database for easy filtering (including chunk_text, chunk_type (real, synthetic), page_number, chunk_type (text, table, image), chunk)
- [ ] Hardcode outputs for some dependent steps to stub to move forward with project
- [ ] Refractor to save embeddings to vector store with partition (collection > partition > entity(embedding))
- [ ] Discuss tool choices (celery +redis/rabbitmq, etc)

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
