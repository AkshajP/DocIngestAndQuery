import sys
import os
import logging
import random
import numpy as np
from typing import List, Dict, Any

# Add parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.vector_store.milvus_client import MilvusClient, create_vector_entity
from db.vector_store.schemas import VectorSearchParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vector_store_example")

# Helper function to generate random embeddings
def generate_random_embedding(dim: int = 768) -> List[float]:
    """Generate a random embedding vector"""
    embedding = np.random.normal(0, 1, dim).tolist()
    # Normalize to unit length
    norm = np.linalg.norm(embedding)
    return [x / norm for x in embedding]

def generate_sample_entities(num_docs: int = 2, chunks_per_doc: int = 5) -> List[Dict[str, Any]]:
    """Generate sample entities for testing"""
    entities = []
    
    # Define some content types
    content_types = ["text", "table", "image"]
    chunk_types = ["original", "summary"]
    
    # Define a few sample case IDs
    case_ids = ["case_A", "case_B", "case_C"]
    
    for doc_idx in range(num_docs):
        document_id = f"doc_{doc_idx + 1}"
        # Assign a case_id (cycling through the available ones)
        case_id = case_ids[doc_idx % len(case_ids)]
        
        # Generate different types of chunks for each document
        for chunk_idx in range(chunks_per_doc):
            chunk_id = f"chunk_{chunk_idx + 1}"
            
            # Vary content by type
            content_type = random.choice(content_types)
            
            if content_type == "text":
                content = f"This is sample text content for document {doc_idx + 1}, chunk {chunk_idx + 1}."
            elif content_type == "table":
                content = f"Sample table content for document {doc_idx + 1}, chunk {chunk_idx + 1}."
            else:  # image
                content = f"Image caption for document {doc_idx + 1}, chunk {chunk_idx + 1}."
            
            # Determine if this is an original chunk or a summary
            chunk_type = chunk_types[0] if chunk_idx < chunks_per_doc - 1 else chunk_types[1]
            
            # Set tree level (0 for original chunks, > 0 for summaries)
            tree_level = 0 if chunk_type == "original" else 1
            
            # Create vector entity
            entity = create_vector_entity(
                document_id=document_id,
                chunk_id=chunk_id,
                content=content,
                embedding=generate_random_embedding(),
                case_id=case_id,  # Add the case_id here
                content_type=content_type,
                chunk_type=chunk_type,
                page_number=random.randint(1, 10),
                tree_level=tree_level,
                metadata={
                    "source": f"example_{doc_idx + 1}",
                    "importance": random.randint(1, 5)
                }
            )
            
            entities.append(entity)
    
    return entities

def main():
    """Example of using the enhanced Milvus vector store client"""
    logger.info("Starting vector store example")
    
    # Initialize Milvus client with a test collection
    test_collection = f"test_collection_{int(random.random() * 10000)}"
    client = MilvusClient(
        host="localhost",
        port="19530",
        collection_name=test_collection,
        dimension=768
    )
    
    try:
        # Generate sample entities
        logger.info("Generating sample entities")
        entities = generate_sample_entities(num_docs=3, chunks_per_doc=5)
        logger.info(f"Generated {len(entities)} sample entities")
        
        # Insert entities into the vector store
        logger.info("Adding entities to vector store")
        success = client.add_entities(entities)
        
        if success:
            logger.info("Entities added successfully")
            
            # Wait a moment for Milvus to fully process (optional)
            import time
            time.sleep(1)
            
            # Get entity count
            count = client.count_entities()
            logger.info(f"Total entities in collection: {count}")
            
            # Get count by content type
            text_count = client.count_entities('content_type == "text"')
            logger.info(f"Text entities: {text_count}")
            
            # Get count by case_id
            case_a_count = client.count_entities('case_id == "case_A"')
            logger.info(f"Case A entities: {case_a_count}")
            
            # Get partition statistics
            partition_stats = client.get_partition_statistics()
            logger.info(f"Partition statistics: {partition_stats}")
            
            # Perform a search
            logger.info("Performing vector search")
            query_embedding = generate_random_embedding()
            
            # Create search parameters with no document filtering first
            search_params = VectorSearchParams(
                query_embedding=query_embedding,
                top_k=3
            )
            
            # Execute search
            results = client.search(search_params)
            logger.info(f"General search returned {len(results)} results")
            
            # Now try with case_id filtering
            case_filtered_search_params = VectorSearchParams(
                query_embedding=query_embedding,
                case_ids=["case_A"],
                top_k=3
            )
            
            # Execute case-filtered search
            case_filtered_results = client.search(case_filtered_search_params)
            
            # Display case-filtered results
            logger.info(f"Case-filtered search returned {len(case_filtered_results)} results")
            for i, result in enumerate(case_filtered_results):
                logger.info(f"Result {i+1}: Score={result.score:.4f}")
                logger.info(f"  - Document: {result.entity.document_id}")
                logger.info(f"  - Case ID: {result.entity.case_id}")
                logger.info(f"  - Content: {result.entity.content[:50]}...")
                logger.info(f"  - Type: {result.entity.content_type}")
                logger.info(f"  - Metadata: {result.entity.metadata}")
            
            # Test combined filtering (document_id and case_id)
            combined_search_params = VectorSearchParams(
                query_embedding=query_embedding,
                document_ids=["doc_1"],
                case_ids=["case_A"],
                content_types=["text"],
                top_k=3
            )
            
            # Execute combined filtered search
            combined_results = client.search(combined_search_params)
            logger.info(f"Combined filtering search returned {len(combined_results)} results")
            
            # Test deletion by case ID
            delete_case_id = "case_B"
            logger.info(f"Deleting entities for case ID: {delete_case_id}")
            client.delete_by_filter(f'case_id == "{delete_case_id}"')
            
            # Wait a moment for deletion to complete
            time.sleep(1)
            
            # Verify deletion
            count_after = client.count_entities()
            logger.info(f"Total entities after deletion: {count_after}")
            
            # Count remaining case_B entities specifically
            case_b_count = client.count_entities('case_id == "case_B"')
            logger.info(f"Remaining case_B entities: {case_b_count}")
            
        else:
            logger.error("Failed to add entities")
        
    except Exception as e:
        logger.error(f"Error in vector store example: {e}")
        # Print a more detailed stack trace to help debug
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up - drop the test collection
        logger.info(f"Cleaning up: dropping collection {test_collection}")
        client.drop()

if __name__ == "__main__":
    main()