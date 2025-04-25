import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from core.config import get_config
from db.document_store.repository import DocumentMetadataRepository
from db.vector_store.adapter import VectorStoreAdapter
from services.document.storage import LocalStorageAdapter, S3StorageAdapter
from services.document.chunker import Chunker
from services.pdf.extractor import PDFExtractor
from services.ml.embeddings import EmbeddingService
from services.retrieval.raptor import Raptor

logger = logging.getLogger(__name__)

def upload_document(
    file_path: str,
    document_id: Optional[str] = None,
    case_id: str = "default",
    metadata: Optional[Dict[str, Any]] = None,
    config = None
) -> Dict[str, Any]:
    """
    Upload and process a document through the complete pipeline.
    
    Args:
        file_path: Path to the document file
        document_id: Optional custom ID (generated if not provided)
        case_id: Case ID for document grouping (required)
        metadata: Optional metadata about the document
        config: Optional config override
        
    Returns:
        Dictionary with processing status and information
    """
    logger.info("First log")
    # Setup components
    config = config or get_config()
    process_start_time = time.time()
    
    # Initialize services
    storage_adapter = _initialize_storage_adapter(config)
    doc_repository = DocumentMetadataRepository()
    vector_store = VectorStoreAdapter(config=config)
    pdf_extractor = PDFExtractor(language=config.processing.language)
    chunker = Chunker(
        max_chunk_size=config.processing.chunk_size,
        min_chunk_size=200,
        overlap_size=config.processing.chunk_overlap
    )
    embedding_service = EmbeddingService(
        model_name=config.ollama.embed_model,
        base_url=config.ollama.base_url
    )
    raptor = Raptor(
        ollama_base_url=config.ollama.base_url,
        ollama_model=config.ollama.model,
        ollama_embed_model=config.ollama.embed_model,
        max_tree_levels=config.processing.max_tree_levels
    )
    
    # Generate document ID if not provided
    if not document_id:
        timestamp = int(time.time())
        base_name = os.path.basename(file_path)
        safe_name = ''.join(c for c in base_name.split('.')[0].replace(' ', '_') 
                          if c.isalnum() or c == '_')
        document_id = f"doc_{timestamp}_{safe_name}"
    
    # Initialize metadata
    doc_metadata = {
        "document_id": document_id,
        "case_id": case_id,
        "original_filename": os.path.basename(file_path),
        "original_file_path": file_path,
        "file_type": os.path.splitext(file_path)[1].lower()[1:],
        "processing_start_time": datetime.now().isoformat(),
        "status": "processing",
        "user_metadata": metadata or {}
    }
    
    # Save initial metadata
    doc_repository.add_document(doc_metadata)
    
    try:
        # Step 1: Extract content from document
        extraction_start_time = time.time()
        logger.info(f"Extracting content from {file_path}")
        extraction_result = pdf_extractor.extract_content(file_path)
        
        if extraction_result["status"] != "success":
            return _handle_processing_failure(
                doc_repository, 
                document_id, 
                "extraction", 
                extraction_result.get("message", "Unknown extraction error")
            )
        
        content_list = extraction_result["content_list"]
        page_count = extraction_result.get("page_count", 0)
        
        # Update metadata with extraction info
        doc_repository.update_document(document_id, {
            "page_count": page_count,
            "extraction_time": time.time() - extraction_start_time
        })
        
        # Step 2: Chunk the extracted content
        chunking_start_time = time.time()
        logger.info(f"Chunking document content for {document_id}")
        chunks = chunker.chunk_content(content_list)
        
        if not chunks:
            return _handle_processing_failure(
                doc_repository,
                document_id,
                "chunking",
                "No chunks created from document content"
            )
        
        # Count content types
        content_types = {}
        for chunk in chunks:
            chunk_type = chunk.get("metadata", {}).get("type", "unknown")
            if chunk_type not in content_types:
                content_types[chunk_type] = 0
            content_types[chunk_type] += 1
        
        chunking_time = time.time() - chunking_start_time
        
        # Update metadata with chunking info
        doc_repository.update_document(document_id, {
            "chunks_count": len(chunks),
            "content_types": content_types,
            "chunking_time": chunking_time
        })
        
        # Step 3: Generate embeddings for chunks
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embedding_start_time = time.time()
        # Prepare texts dictionary with chunk IDs and contents
        texts_dict = {chunk["id"]: chunk["content"] for chunk in chunks}
        
        # Generate embeddings
        chunk_embeddings = embedding_service.generate_embeddings_dict(
            texts_dict, 
            show_progress=True
        )
        
        if not chunk_embeddings:
            return _handle_processing_failure(
                doc_repository,
                document_id,
                "embedding",
                "Failed to generate embeddings"
            )
        
        embedding_time = time.time() - embedding_start_time
        
        # Update metadata with embedding info
        doc_repository.update_document(document_id, {
            "embedding_time": embedding_time
        })
        
        # Step 4: Build RAPTOR tree
        raptor_start_time = time.time()
        logger.info(f"Building RAPTOR tree for document {document_id}")
        
        # Format chunks for RAPTOR
        chunk_texts = [chunk["content"] for chunk in chunks]
        chunk_ids = [chunk["id"] for chunk in chunks]
        
        # Pass to RAPTOR for tree building
        tree_data = raptor.build_tree(chunk_texts, chunk_ids, chunk_embeddings)
        
        if not tree_data:
            return _handle_processing_failure(
                doc_repository,
                document_id,
                "tree_building",
                "Failed to build RAPTOR tree"
            )
        
        # Generate embeddings for tree nodes
        tree_nodes = {}
        for level, level_data in tree_data.items():
            if level <= 0:
                continue
                
            summaries_df = level_data.get("summaries_df")
            if summaries_df is None:
                continue
                
            for _, row in summaries_df.iterrows():
                summary_text = row.get("summaries", "")
                cluster_id = row.get("cluster")
                
                if not summary_text:
                    continue
                
                # Create node ID
                node_id = f"summary_l{level}_c{cluster_id}"
                tree_nodes[node_id] = summary_text
        
        # Generate embeddings for tree nodes
        tree_embeddings = embedding_service.generate_embeddings_dict(
            tree_nodes, 
            show_progress=True
        )
        
        tree_build_time = time.time() - raptor_start_time
        
        doc_repository.update_document(document_id, {
            "raptor_tree_build_time": tree_build_time
        })
        # Step 5: Store document chunks and tree in vector database
        logger.info(f"Storing document chunks and tree nodes in vector database")
        
        # Store original chunks
        chunks_count = vector_store.add_document_chunks(
            document_id, 
            case_id, 
            chunks, 
            chunk_embeddings
        )
        
        if chunks_count == 0:
            return _handle_processing_failure(
                doc_repository,
                document_id,
                "vector_storage",
                "Failed to store document chunks in vector database"
            )
        
        # Store tree nodes
        nodes_count = vector_store.add_tree_nodes(
            document_id, 
            case_id, 
            tree_data, 
            tree_embeddings
        )
        
        # Calculate total processing time
        total_processing_time = time.time() - process_start_time
        
        # Update document metadata with success status
        raptor_levels = sorted(list(tree_data.keys()))
        doc_repository.update_document(document_id, {
            "status": "processed",
            "total_processing_time": total_processing_time,
            "raptor_levels": raptor_levels,
            "processing_date": datetime.now().isoformat()
        })
        
        logger.info(f"Document {document_id} processed successfully in {total_processing_time:.2f} seconds")
        
        return {
            "status": "success",
            "document_id": document_id,
            "case_id": case_id,
            "processing_time": total_processing_time,
            "chunks_count": chunks_count,
            "tree_nodes_count": nodes_count,
            "raptor_levels": raptor_levels
        }
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        return _handle_processing_failure(
            doc_repository, 
            document_id, 
            "processing", 
            str(e)
        )

def _initialize_storage_adapter(config):
    """Initialize the appropriate storage adapter based on configuration"""
    storage_type = config.storage.storage_type.lower()
    
    if storage_type == "s3":
        logger.info(f"Initializing S3 storage adapter with bucket {config.storage.s3_bucket}")
        return S3StorageAdapter(
            bucket_name=config.storage.s3_bucket,
            prefix=config.storage.s3_prefix,
            region=config.storage.aws_region
        )
    else:
        logger.info(f"Initializing local storage adapter with directory {config.storage.storage_dir}")
        return LocalStorageAdapter()

def _handle_processing_failure(
    doc_repository,
    document_id: str, 
    stage: str, 
    error_message: str
) -> Dict[str, Any]:
    """
    Handle document processing failure.
    
    Args:
        doc_repository: Document metadata repository
        document_id: Document ID
        stage: Processing stage that failed
        error_message: Error message
        
    Returns:
        Error response dictionary
    """
    failure_metadata = {
        "status": "failed",
        "failure_stage": stage,
        "error_message": error_message,
        "failure_time": datetime.now().isoformat()
    }
    
    # Update document metadata
    doc_repository.update_document(document_id, failure_metadata)
    
    logger.error(f"Document {document_id} processing failed at {stage}: {error_message}")
    
    return {
        "status": "error",
        "document_id": document_id,
        "stage": stage,
        "error": error_message
    }