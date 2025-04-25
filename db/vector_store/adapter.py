import logging
from typing import List, Dict, Any, Optional, Union
import uuid
import hashlib

from .milvus_client import MilvusClient, create_vector_entity
from .schemas import VectorEntity, VectorSearchParams, VectorSearchResult
from core.config import get_config

logger = logging.getLogger(__name__)

class VectorStoreAdapter:
    """
    Adapter for vector database operations.
    Abstracts the implementation details of the vector database.
    """
    
    def __init__(
        self,
        config=None,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        """
        Initialize the vector store adapter.
        
        Args:
            config: Configuration object (if None, will load from global config)
            collection_name: Override collection name from config
            dimension: Override embedding dimension from config
        """
        self.config = config or get_config()
        self.vector_db_config = self.config.vector_db
        
        # Allow overriding collection name and dimension
        self.collection_name = collection_name or self.vector_db_config.collection_name
        self.dimension = dimension or self.vector_db_config.dimension
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the vector database client"""
        try:
            self.client = MilvusClient(
                host=self.vector_db_config.host,
                port=self.vector_db_config.port,
                user=self.vector_db_config.username or "",
                password=self.vector_db_config.password or "",
                collection_name=self.collection_name,
                dimension=self.dimension
            )
            logger.info(f"Initialized vector store client for collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store client: {str(e)}")
            raise
    
    def add_document_chunks(
        self, 
        document_id: str, 
        case_id: str,
        chunks: List[Dict[str, Any]], 
        embeddings: Dict[str, List[float]]
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            document_id: Document ID
            case_id: Case ID for grouping documents (required)
            chunks: List of chunk objects with content and metadata
            embeddings: Dictionary of chunk_id to embedding vector
            
        Returns:
            Number of chunks successfully added
        """
        if not chunks:
            logger.warning(f"No chunks provided for document {document_id}")
            return 0
        
        entities = []
        skipped_chunks = 0
        
        for chunk in chunks:
            chunk_id = chunk["id"]
            content = chunk["content"]
            metadata = chunk.get("metadata", {})
            
            # Get embedding for this chunk
            if chunk_id not in embeddings:
                logger.warning(f"No embedding found for chunk {chunk_id} in document {document_id}")
                skipped_chunks += 1
                continue
            
            embedding = embeddings[chunk_id]
            
            # Check if embedding is valid
            if not embedding or not isinstance(embedding, list):
                logger.warning(f"Invalid embedding for chunk {chunk_id} in document {document_id}")
                skipped_chunks += 1
                continue
            
            # Determine content type and page number
            content_type = metadata.get("type", "text")
            page_number = metadata.get("page_idx")
            
            # Preserve original bounding boxes if available
            original_boxes = metadata.get("original_boxes", [])
            
            # Log some details about this chunk
            logger.debug(f"Processing chunk {chunk_id}, type: {content_type}, embedding length: {len(embedding)}")
            
            # Create entity
            try:
                entity = create_vector_entity(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    content=content,
                    embedding=embedding,
                    case_id=case_id,  # Required case_id
                    content_type=content_type,
                    chunk_type="original",  # Original document chunk
                    page_number=page_number,
                    tree_level=0,  # Base level for original chunks
                    metadata={
                        **metadata,
                        "original_boxes": original_boxes
                    }
                )
                
                entities.append(entity)
            except Exception as e:
                logger.error(f"Error creating entity for chunk {chunk_id}: {str(e)}")
                skipped_chunks += 1
        
        if not entities:
            logger.error(f"No valid entities created for document {document_id}")
            return 0
        
        # Add entities to vector store
        logger.info(f"Adding {len(entities)} entities to vector store for document {document_id}")
        success = self.client.add_entities(entities)
        
        if not success:
            logger.error(f"Failed to add chunks for document {document_id}")
            return 0
        
        logger.info(f"Successfully added {len(entities)} chunks for document {document_id} (skipped {skipped_chunks})")
        return len(entities)
    
    def add_tree_nodes(
        self,
        document_id: str,
        case_id: str,
        tree_data: Dict[int, Dict[str, Any]],
        tree_embeddings: Dict[str, List[float]]
    ) -> int:
        """
        Add RAPTOR tree nodes to the vector store.
        
        Args:
            document_id: Document ID
            case_id: Case ID for grouping documents (required)
            tree_data: Dictionary mapping levels to cluster and summary data
            tree_embeddings: Dictionary of node IDs to embeddings
            
        Returns:
            Number of nodes successfully added
        """
        entities = []
        
        for level, level_data in tree_data.items():
            # Skip level 0 as those are original chunks
            if level <= 0:
                continue
                
            summaries_df = level_data.get("summaries_df")
            if summaries_df is None:
                continue
                
            # Process each summary
            for _, row in summaries_df.iterrows():
                summary_text = row.get("summaries", "")
                cluster_id = row.get("cluster")
                
                if not summary_text:
                    continue
                
                # Create node ID
                node_id = f"summary_l{level}_c{cluster_id}"
                
                # Get embedding
                if node_id not in tree_embeddings:
                    logger.warning(f"No embedding found for tree node {node_id}")
                    continue
                
                embedding = tree_embeddings[node_id]
                
                # Create entity
                entity = create_vector_entity(
                    document_id=document_id,
                    chunk_id=node_id,
                    content=summary_text,
                    embedding=embedding,
                    case_id=case_id,  # Required case_id
                    content_type="text",
                    chunk_type="summary",
                    page_number=None,
                    tree_level=level,
                    metadata={
                        "cluster": cluster_id
                    }
                )
                
                entities.append(entity)
        
        # Add entities to vector store
        if not entities:
            logger.warning(f"No tree nodes to add for document {document_id}")
            return 0
            
        success = self.client.add_entities(entities)
        
        if not success:
            logger.error(f"Failed to add tree nodes for document {document_id}")
            return 0
        
        return len(entities)
    
    def search(
        self,
        query_embedding: List[float],
        case_ids: List[str],
        document_ids: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None,
        tree_levels: Optional[List[int]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding vector
            case_ids: List of case IDs to filter by (required)
            document_ids: Optional list of document IDs to filter by
            content_types: Optional list of content types to filter by
            chunk_types: Optional list of chunk types to filter by
            tree_levels: Optional list of tree levels to filter by
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with content and metadata
        """
        # Create search parameters
        search_params = VectorSearchParams(
            query_embedding=query_embedding,
            case_ids=case_ids,  # Required case_ids
            document_ids=document_ids,
            content_types=content_types,
            chunk_types=chunk_types,
            tree_levels=tree_levels,
            top_k=top_k
        )
        
        # Execute search
        results = self.client.search(search_params)
        
        # Format results
        formatted_results = []
        
        for result in results:
            entity = result.entity
            
            formatted_result = {
                "document_id": entity.document_id,
                "case_id": entity.case_id,
                "content": entity.content,
                "score": result.score,
                "metadata": entity.metadata,
                "content_type": entity.content_type,
                "chunk_type": entity.chunk_type,
                "page_number": entity.page_number,
                "tree_level": entity.tree_level
            }
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def delete_document(self, document_id: str, case_id: str) -> bool:
        """
        Delete all vectors for a document in a specific case.
        
        Args:
            document_id: Document ID to delete
            case_id: Case ID the document belongs to (required)
            
        Returns:
            True if successful, False otherwise
        """
        return self.client.delete_by_filter(f'document_id == "{document_id}" && case_id == "{case_id}"')
    
    def delete_documents(self, document_ids: List[str], case_id: str) -> bool:
        """
        Delete all vectors for multiple documents in a specific case.
        
        Args:
            document_ids: List of document IDs to delete
            case_id: Case ID the documents belong to (required)
            
        Returns:
            True if successful, False otherwise
        """
        if len(document_ids) == 1:
            return self.delete_document(document_ids[0], case_id)
            
        docs_str = '", "'.join(document_ids)
        return self.client.delete_by_filter(f'document_id in ["{docs_str}"] && case_id == "{case_id}"')
    
    def count_document_chunks(self, document_id: str, case_id: str) -> int:
        """
        Count the number of chunks for a document in a specific case.
        
        Args:
            document_id: Document ID
            case_id: Case ID the document belongs to (required)
            
        Returns:
            Number of chunks
        """
        return self.client.count_entities(f'document_id == "{document_id}" && case_id == "{case_id}"')
    
    def release(self):
        """Release resources"""
        if hasattr(self, "client"):
            self.client.release()