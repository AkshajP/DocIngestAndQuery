# migration_utility.py
import logging
import time
from typing import List, Dict, Any, Optional

from db.vector_store.adapter import VectorStoreAdapter
from db.document_store.repository import DocumentMetadataRepository
from core.config import get_config

logger = logging.getLogger(__name__)

def migrate_collection(
    source_collection_name: str = "document_store",
    target_collection_name: str = "document_store_hybrid",
    batch_size: int = 100,
    max_documents: Optional[int] = None,
    document_ids: Optional[List[str]] = None
):
    """
    Migrate data from existing collection to a new hybrid-enabled collection.
    
    Args:
        source_collection_name: Name of the source collection
        target_collection_name: Name of the target collection with BM25 support
        batch_size: Number of documents to process in one batch
        max_documents: Maximum number of documents to migrate (None = all)
        document_ids: Specific document IDs to migrate (None = all)
    
    Returns:
        Dictionary with migration statistics
    """
    config = get_config()
    stats = {"start_time": time.time(), "documents_processed": 0, "chunks_migrated": 0, "errors": 0}
    
    # Initialize adapters for source and target collections
    source_adapter = VectorStoreAdapter(
        config=config,
        collection_name=source_collection_name,
        hybrid_search_enabled=False
    )
    
    target_adapter = VectorStoreAdapter(
        config=config,
        collection_name=target_collection_name,
        hybrid_search_enabled=True
    )
    
    # Get document metadata repository
    doc_repo = DocumentMetadataRepository()
    
    try:
        # Get list of documents to migrate
        all_documents = doc_repo.list_documents()
        
        # Filter documents if specific IDs are provided
        if document_ids:
            filtered_docs = [doc for doc in all_documents if doc.get("document_id") in document_ids]
            documents_to_migrate = filtered_docs
        else:
            documents_to_migrate = all_documents
        
        # Apply max_documents limit if specified
        if max_documents is not None:
            documents_to_migrate = documents_to_migrate[:max_documents]
        
        total_documents = len(documents_to_migrate)
        logger.info(f"Starting migration of {total_documents} documents")
        
        # Process documents in batches
        for i in range(0, total_documents, batch_size):
            batch = documents_to_migrate[i:i+batch_size]
            batch_doc_ids = [doc.get("document_id") for doc in batch]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_documents-1)//batch_size + 1} with {len(batch)} documents")
            
            for doc in batch:
                doc_id = doc.get("document_id")
                case_id = doc.get("case_id", "default")
                
                try:
                    # Get all vector entries for this document from source collection
                    # This requires implementing a method to extract documents with embeddings
                    vectors = _extract_document_vectors(source_adapter, doc_id, case_id)
                    
                    if vectors:
                        # Add vectors to target collection
                        result = _import_vectors_to_target(target_adapter, vectors)
                        
                        stats["chunks_migrated"] += result.get("chunks_migrated", 0)
                        if result.get("error"):
                            stats["errors"] += 1
                            logger.error(f"Error migrating document {doc_id}: {result['error']}")
                    
                    stats["documents_processed"] += 1
                    logger.info(f"Migrated document {doc_id} with {len(vectors)} chunks")
                    
                except Exception as e:
                    stats["errors"] += 1
                    logger.error(f"Error processing document {doc_id}: {str(e)}")
        
        stats["end_time"] = time.time()
        stats["total_time"] = stats["end_time"] - stats["start_time"]
        
        logger.info(f"Migration completed: {stats['documents_processed']} documents, " +
                   f"{stats['chunks_migrated']} chunks, {stats['errors']} errors, " +
                   f"in {stats['total_time']:.2f} seconds")
        
        return stats
    
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        stats["error"] = str(e)
        return stats

def _extract_document_vectors(
    adapter: VectorStoreAdapter,
    document_id: str,
    case_id: str
) -> List[Dict[str, Any]]:
    """
    Extract document vectors from source collection.
    
    This is challenging because we need to get vectors with their embeddings,
    which typically aren't returned in search results.
    
    Args:
        adapter: Source VectorStoreAdapter
        document_id: Document ID to extract
        case_id: Case ID for the document
        
    Returns:
        List of vectors with embeddings and metadata
    """
    # Need to implement a new method in MilvusClient to extract vectors with embeddings
    # This might require direct Milvus querying with output_fields including "embedding"
    
    if hasattr(adapter.client, 'extract_document_vectors'):
        # Call the method if available
        return adapter.client.extract_document_vectors(document_id, case_id)
    else:
        # Fall back to manual implementation
        logger.warning(f"extract_document_vectors not implemented in client, using fallback")
        
        # This requires direct access to Milvus
        collection = adapter.client.collection
        
        # Query for vectors with embeddings
        expr = f'document_id == "{document_id}" && case_id == "{case_id}"'
        results = collection.query(
            expr=expr,
            output_fields=["id", "document_id", "case_id", "chunk_id", "content", 
                          "content_type", "chunk_type", "page_number", "tree_level", 
                          "metadata", "embedding"]
        )
        
        # Convert to vector entity format
        vectors = []
        for result in results:
            vector = {
                "id": result.get("id"),
                "document_id": result.get("document_id"),
                "case_id": result.get("case_id", "default"),
                "chunk_id": result.get("chunk_id"),
                "content": result.get("content"),
                "content_type": result.get("content_type", "text"),
                "chunk_type": result.get("chunk_type", "original"),
                "page_number": result.get("page_number"),
                "tree_level": result.get("tree_level", 0),
                "metadata": result.get("metadata", {}),
                "embedding": result.get("embedding", [])
            }
            vectors.append(vector)
        
        return vectors

def _import_vectors_to_target(
    target_adapter: VectorStoreAdapter,
    vectors: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Import vectors to target collection.
    
    Args:
        target_adapter: Target VectorStoreAdapter
        vectors: List of vector entities to import
        
    Returns:
        Dictionary with import statistics
    """
    stats = {"chunks_migrated": 0, "error": None}
    
    try:
        # Convert dict to VectorEntity objects
        from db.vector_store.schemas import VectorEntity
        entities = []
        
        for vector in vectors:
            entity = VectorEntity(**vector)
            entities.append(entity)
        
        # Add to target collection
        success = target_adapter.client.add_entities(entities)
        
        if success:
            stats["chunks_migrated"] = len(entities)
        else:
            stats["error"] = "Failed to add entities to target collection"
        
        return stats
    
    except Exception as e:
        stats["error"] = str(e)
        return stats