#!/usr/bin/env python
"""
Diagnostic script for analyzing and fixing issues with document processing.
This script helps identify common problems with embeddings, document chunks, and Milvus.
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional

# Add parent directory to import path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import required modules
from core.config import get_config
from services.ml.embeddings import EmbeddingService
from db.vector_store.milvus_client import MilvusClient
from services.document.chunker import Chunker
from services.pdf.extractor import PDFExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("diagnostics")

def test_milvus_connection():
    """Test connection to Milvus and print diagnostics"""
    print("\n--- Testing Milvus Connection ---")
    
    config = get_config()
    try:
        client = MilvusClient(
            host=config.vector_db.host,
            port=config.vector_db.port,
            collection_name=config.vector_db.collection_name,
            dimension=config.vector_db.dimension
        )
        
        print(f"✅ Successfully connected to Milvus at {config.vector_db.host}:{config.vector_db.port}")
        print(f"Collection: {config.vector_db.collection_name}")
        print(f"Expected dimension: {config.vector_db.dimension}")
        
        # Try to get entity count
        try:
            count = client.count_entities()
            print(f"Total entities: {count}")
        except Exception as e:
            print(f"⚠️ Could not count entities: {str(e)}")
        
        # Get partition info
        try:
            partition_stats = client.get_partition_statistics()
            print(f"Partition statistics: {partition_stats}")
        except Exception as e:
            print(f"⚠️ Could not get partition statistics: {str(e)}")
            
        # Test vector insertion
        try:
            test_entity = {
                "id": "test_entity",
                "document_id": "test_doc",
                "case_id": "test_case",
                "chunk_id": "test_chunk",
                "content": "This is a test entity for diagnostics",
                "content_type": "text",
                "chunk_type": "original",
                "page_number": 1,
                "tree_level": 0,
                "metadata": {"test": True},
                "embedding": [0.1] * config.vector_db.dimension  # Generate test embedding
            }
            
            # Convert to VectorEntity
            from db.vector_store.schemas import VectorEntity
            vector_entity = VectorEntity(**test_entity)
            
            # Try to insert
            success = client.add_entities([vector_entity])
            
            if success:
                print("✅ Successfully inserted test entity")
                # Clean up
                client.delete_by_filter('id == "test_entity"')
            else:
                print("❌ Failed to insert test entity")
        except Exception as e:
            print(f"❌ Error testing vector insertion: {str(e)}")
            
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {str(e)}")

def test_embedding_service():
    """Test the embedding service and print statistics"""
    print("\n--- Testing Embedding Service ---")
    
    config = get_config()
    try:
        embedding_service = EmbeddingService(
            model_name=config.ollama.embed_model,
            base_url=config.ollama.base_url,
            default_dimension=config.vector_db.dimension
        )
        
        print(f"✅ Successfully initialized embedding service")
        print(f"Model: {config.ollama.embed_model}")
        print(f"Dimension: {embedding_service.embedding_dim}")
        
        # Generate test embeddings
        test_texts = [
            "This is a short test text for embedding.",
            "Another example with different content to test the embedding service.",
            "A third text sample that will help diagnose any potential issues."
        ]
        
        print("\nGenerating test embeddings...")
        embeddings = embedding_service.generate_embeddings_batch(test_texts)
        
        if embeddings:
            print(f"✅ Generated {len(embeddings)} embeddings")
            for i, emb in enumerate(embeddings):
                print(f"Embedding {i+1}: dimension={len(emb)}")
                
            # Check for NaN or Inf values
            for i, emb in enumerate(embeddings):
                if any(np.isnan(x) for x in emb) or any(np.isinf(x) for x in emb):
                    print(f"⚠️ Embedding {i+1} contains NaN or Inf values!")
        else:
            print("❌ Failed to generate embeddings")
            
    except Exception as e:
        print(f"❌ Error testing embedding service: {str(e)}")

def test_chunking(pdf_path=None):
    """Test document chunking with a sample PDF"""
    print("\n--- Testing Document Chunking ---")
    
    if not pdf_path or not os.path.exists(pdf_path):
        print("⚠️ No valid PDF path provided, skipping chunking test")
        return
    
    config = get_config()
    try:
        # Initialize components
        pdf_extractor = PDFExtractor(language=config.processing.language)
        chunker = Chunker(
            max_chunk_size=config.processing.chunk_size,
            min_chunk_size=200,
            overlap_size=config.processing.chunk_overlap
        )
        
        print(f"Extracting content from {pdf_path}...")
        extraction_result = pdf_extractor.extract_content(pdf_path)
        
        if extraction_result["status"] != "success":
            print(f"❌ Extraction failed: {extraction_result.get('message', 'Unknown error')}")
            return
            
        content_list = extraction_result["content_list"]
        print(f"✅ Extracted {len(content_list)} content items")
        
        # Count types
        content_types = {}
        for item in content_list:
            content_type = item.get("type", "unknown")
            if content_type not in content_types:
                content_types[content_type] = 0
            content_types[content_type] += 1
        
        print(f"Content types: {content_types}")
        
        # Test chunking
        print("Chunking document content...")
        chunks = chunker.chunk_content(content_list)
        
        if not chunks:
            print("❌ No chunks created")
            return
            
        print(f"✅ Created {len(chunks)} chunks")
        
        # Count chunk types
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk.get("metadata", {}).get("type", "unknown")
            if chunk_type not in chunk_types:
                chunk_types[chunk_type] = 0
            chunk_types[chunk_type] += 1
        
        print(f"Chunk types: {chunk_types}")
        
        # Print first chunk details
        if chunks:
            first_chunk = chunks[0]
            print("\nSample chunk:")
            print(f"ID: {first_chunk['id']}")
            print(f"Type: {first_chunk.get('metadata', {}).get('type', 'unknown')}")
            print(f"Content length: {len(first_chunk['content'])} chars")
            
    except Exception as e:
        print(f"❌ Error testing chunking: {str(e)}")

def run_all_tests(pdf_path=None):
    """Run all diagnostic tests"""
    print("\n" + "="*50)
    print("      DOCUMENT PROCESSING DIAGNOSTICS")
    print("="*50)
    
    # Load configuration
    config = get_config()
    print("\n--- Configuration ---")
    print(f"Vector DB: {config.vector_db.host}:{config.vector_db.port}")
    print(f"Collection: {config.vector_db.collection_name}")
    print(f"Dimension: {config.vector_db.dimension}")
    print(f"Ollama: {config.ollama.base_url}")
    print(f"Embed model: {config.ollama.embed_model}")
    
    # Run tests
    test_milvus_connection()
    test_embedding_service()
    
    if pdf_path:
        test_chunking(pdf_path)
    
    print("\n" + "="*50)
    print("      DIAGNOSTICS COMPLETE")
    print("="*50)

if __name__ == "__main__":
    # Check if PDF path was provided
    pdf_path = None
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if not os.path.exists(pdf_path):
            print(f"Warning: Provided PDF path does not exist: {pdf_path}")
            pdf_path = None
    
    run_all_tests(pdf_path)