#!/usr/bin/env python
"""
Migration script to rebuild the vector store with the correct dimension.
This will drop the existing collection and create a new one with 3072 dimensions.
"""

import os
import sys
import logging
import argparse
from typing import List, Dict, Any

# Add parent directory to import path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import required modules
from pymilvus import connections, utility
from core.config import get_config, reset_config
from db.vector_store.milvus_client import MilvusClient
from db.document_store.repository import DocumentMetadataRepository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("migration")

def rebuild_collection(confirm: bool = False, dimension: int = 3072):
    """
    Rebuild the vector store collection with the correct dimension.
    
    Args:
        confirm: Whether to skip confirmation prompt
        dimension: New dimension for embeddings
    """
    # Load config
    config = get_config()
    collection_name = config.vector_db.collection_name
    
    # Connect to Milvus
    try:
        connections.connect(
            alias="default",
            host=config.vector_db.host,
            port=config.vector_db.port,
            user=config.vector_db.username or "",
            password=config.vector_db.password or ""
        )
        logger.info(f"Connected to Milvus at {config.vector_db.host}:{config.vector_db.port}")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {str(e)}")
        return False
    
    # Check if collection exists
    if utility.has_collection(collection_name):
        current_dimension = get_current_dimension(collection_name)
        logger.info(f"Found existing collection '{collection_name}' with dimension: {current_dimension}")
        
        if not confirm:
            response = input(f"This will DROP the existing collection '{collection_name}' and recreate it with dimension {dimension}. Continue? (y/N): ")
            if response.lower() != 'y':
                logger.info("Operation cancelled")
                return False
        
        # Drop the collection
        try:
            utility.drop_collection(collection_name)
            logger.info(f"Dropped collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to drop collection: {str(e)}")
            return False
    
    # Create new collection with correct dimension
    try:
        # Override the dimension in config
        config.vector_db.dimension = dimension
        
        # Create new client which will initialize the collection
        client = MilvusClient(
            host=config.vector_db.host,
            port=config.vector_db.port,
            collection_name=collection_name,
            dimension=dimension
        )
        
        logger.info(f"Successfully created collection '{collection_name}' with dimension {dimension}")
        
        # Clean up
        client.release()
        
        return True
    except Exception as e:
        logger.error(f"Failed to create collection: {str(e)}")
        return False

def get_current_dimension(collection_name: str) -> int:
    """Get the current dimension of the collection"""
    try:
        # Get collection schema information
        collection_info = utility.get_collection_stats(collection_name)
        
        # Try to extract dimension - this might need adjustment depending on Milvus version
        # For newer Milvus versions:
        from pymilvus import Collection
        coll = Collection(collection_name)
        schema = coll.schema
        
        for field in schema.fields:
            if field.dtype == 101:  # FLOAT_VECTOR type
                return field.params.get("dim")
        
        return 0  # Unknown
    except Exception as e:
        logger.error(f"Failed to get collection dimension: {str(e)}")
        return 0

def mark_documents_for_reprocessing():
    """Mark all documents in the repository for reprocessing"""
    try:
        doc_repo = DocumentMetadataRepository()
        documents = doc_repo.list_documents()
        
        for doc in documents:
            doc_id = doc.get("document_id")
            if doc_id:
                # Mark document as needing reprocessing
                doc_repo.update_document(doc_id, {
                    "status": "pending",
                    "needs_embedding_update": True
                })
                logger.info(f"Marked document {doc_id} for reprocessing")
        
        return len(documents)
    except Exception as e:
        logger.error(f"Failed to mark documents for reprocessing: {str(e)}")
        return 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Migrate vector store to correct dimension")
    parser.add_argument("--dimension", type=int, default=3072, help="New embedding dimension (default: 3072)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--mark-reprocess", action="store_true", help="Mark documents for reprocessing", default=True)
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("      VECTOR STORE MIGRATION: DIMENSION UPDATE")
    print("="*60)
    
    # Reset config to ensure we use latest settings
    reset_config()
    
    # Rebuild collection with new dimension
    success = rebuild_collection(confirm=args.yes, dimension=args.dimension)
    
    if success:
        print("\n✅ Successfully rebuilt vector store with dimension", args.dimension)
        
        if args.mark_reprocess:
            num_docs = mark_documents_for_reprocessing()
            print(f"\n✅ Marked {num_docs} documents for reprocessing")
            print("\nTo reprocess these documents, you'll need to run your document processing script.")
    else:
        print("\n❌ Failed to rebuild vector store")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()