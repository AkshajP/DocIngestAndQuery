import logging
from typing import List, Dict, Any, Optional, Union
import uuid
import hashlib
import time

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
        dimension: Optional[int] = None,
        hybrid_search_enabled: bool = True  # New parameter
    ):
        """
        Initialize the vector store adapter.
        
        Args:
            config: Configuration object (if None, will load from global config)
            collection_name: Override collection name from config
            dimension: Override embedding dimension from config
            hybrid_search_enabled: Whether to enable hybrid search capabilities
        """
        self.config = config or get_config()
        self.vector_db_config = self.config.vector_db
        
        # Allow overriding collection name and dimension
        self.collection_name = collection_name or self.vector_db_config.collection_name
        self.dimension = dimension or self.vector_db_config.dimension
        
        # Store hybrid search configuration
        self.hybrid_search_enabled = hybrid_search_enabled
        
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
                dimension=self.dimension,
                hybrid_search_enabled=self.hybrid_search_enabled  # Pass to client
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
        
        # Process chunks to create entities
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
            
            # Important: Ensure content is properly formatted for BM25 indexing
            # For text content, we might want to apply preprocessing, like:
            # - Making sure it's properly encoded as UTF-8
            # - Ensuring it has no special characters that could cause issues
            # - Limiting length to avoid issues with very long texts
            
            # For BM25 to work effectively, text should be well-formed
            if content_type == "text" and isinstance(content, str):
                # Simple cleaning - remove any control characters that might cause indexing issues
                content = ''.join(ch for ch in content if ch.isprintable() or ch.isspace())
            
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
        MAX_TEXT_LENGTH = 65000  # Safe limit below 65535
        split_nodes_count = 0
        original_nodes_count = 0
        
        # Initialize embedding service for potential new embeddings needed for split nodes
        from services.ml.embeddings import EmbeddingService
        embedding_service = None  # Lazy initialize only if needed
        
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
                
                original_nodes_count += 1
                
                # Create node ID
                node_id = f"summary_l{level}_c{cluster_id}"
                
                # Get embedding
                if node_id not in tree_embeddings:
                    logger.warning(f"No embedding found for tree node {node_id}")
                    continue
                
                embedding = tree_embeddings[node_id]
                
                # Check if summary needs to be split due to length
                if len(summary_text) > MAX_TEXT_LENGTH:
                    logger.info(f"Summary for node {node_id} exceeds max length ({len(summary_text)} chars). Splitting...")
                    
                    # Split into semantic chunks (paragraphs first, then sentences if needed)
                    summary_chunks = self._intelligently_split_summary(summary_text, MAX_TEXT_LENGTH)
                    
                    # Create multiple nodes with index suffixes
                    for i, chunk in enumerate(summary_chunks):
                        # Create unique node ID with chunk index
                        chunk_node_id = f"{node_id}_p{i}"
                        
                        # For first chunk, use the original embedding
                        # For additional chunks, generate new embeddings
                        chunk_embedding = None
                        if i == 0:
                            chunk_embedding = embedding
                        else:
                            # Lazy initialize the embedding service
                            if embedding_service is None:
                                embedding_service = EmbeddingService(
                                    model_name=self.config.ollama.embed_model,
                                    base_url=self.config.ollama.base_url
                                )
                            
                            # Generate embedding for this chunk
                            try:
                                chunk_embedding = embedding_service.generate_embedding(chunk)
                            except Exception as e:
                                logger.error(f"Error generating embedding for split node {chunk_node_id}: {str(e)}")
                                # Fall back to using the original embedding
                                chunk_embedding = embedding
                        
                        # Create entity with reference to other chunks
                        entity = create_vector_entity(
                            document_id=document_id,
                            chunk_id=chunk_node_id,
                            content=chunk,
                            embedding=chunk_embedding,
                            case_id=case_id,
                            content_type="text",
                            chunk_type="summary",
                            page_number=None,
                            tree_level=level,
                            metadata={
                                "cluster": cluster_id,
                                "is_split": True,
                                "split_index": i,
                                "total_splits": len(summary_chunks),
                                "original_node_id": node_id
                            }
                        )
                        
                        entities.append(entity)
                        split_nodes_count += 1
                else:
                    # Regular case - no splitting needed
                    entity = create_vector_entity(
                        document_id=document_id,
                        chunk_id=node_id,
                        content=summary_text,
                        embedding=embedding,
                        case_id=case_id,
                        content_type="text",
                        chunk_type="summary",
                        page_number=None,
                        tree_level=level,
                        metadata={
                            "cluster": cluster_id
                        }
                    )
                    
                    entities.append(entity)
        
        if split_nodes_count > 0:
            logger.info(f"Split {split_nodes_count - original_nodes_count} oversized nodes into {split_nodes_count} semantic chunks")
        
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
        top_k: int = 5,
        use_hybrid: bool = False,
        vector_weight: float = 0.5,
        query_text: Optional[str] = None,
        fusion_method: str = "weighted"  # "weighted" or "rrf"
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors with optional hybrid BM25 search.
        
        Args:
            query_embedding: Query embedding vector
            case_ids: List of case IDs to filter by (required)
            document_ids: Optional list of document IDs to filter by
            content_types: Optional list of content types to filter by
            chunk_types: Optional list of chunk types to filter by
            tree_levels: Optional list of tree levels to filter by
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid search (vector + BM25)
            vector_weight: Weight for vector scores in hybrid search (0-1)
            query_text: Original query text for BM25 (required if use_hybrid=True)
            fusion_method: Method to use for score fusion ("weighted" or "rrf")
            
        Returns:
            List of result dictionaries with content and metadata
        """
        # Input validation
        if use_hybrid and not self.supports_hybrid_search():
            logger.warning("Hybrid search requested but not supported. Falling back to vector search.")
            use_hybrid = False
        
        if use_hybrid and not query_text:
            logger.warning("Hybrid search requested but no query_text provided. Falling back to vector search.")
            use_hybrid = False
        
        # Apply bounds to vector_weight
        vector_weight = max(0.0, min(1.0, vector_weight))
        
        # Create search parameters
        search_params = VectorSearchParams(
            query_embedding=query_embedding,
            case_ids=case_ids,
            document_ids=document_ids,
            content_types=content_types,
            chunk_types=chunk_types,
            tree_levels=tree_levels,
            top_k=top_k
        )
        
        # Execute search based on mode
        search_start = time.time()
        
        try:
            if use_hybrid:
                logger.debug(f"Hybrid search parameters: query_text='{query_text[:50]}...', "
                     f"vector_weight={vector_weight}, fusion={fusion_method}")
                logger.info(f"Executing hybrid search with vector_weight={vector_weight}, fusion={fusion_method}")
                results = self.client.search(
                    search_params=search_params,
                    use_hybrid=True,
                    query_text=query_text,
                    vector_weight=vector_weight,
                    # fusion_method=fusion_method
                )
            else:
                logger.info("Executing vector-only search")
                results = self.client.search(search_params)
            
            search_time = time.time() - search_start
            logger.info(f"Search completed in {search_time:.3f}s with {len(results)} results")
            
            # Format results
            formatted_results = self._format_search_results(results, use_hybrid)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            # Fall back to vector search if hybrid search fails
            if use_hybrid:
                logger.info("Falling back to vector-only search due to error")
                results = self.client.search(search_params)
                return self._format_search_results(results, False)
            else:
                # If already using vector search, just return empty results
                return []
    
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

    def _intelligently_split_summary(self, summary_text: str, max_length: int = 65000) -> List[str]:
        """
        Split a long summary into semantically meaningful chunks that fit within database limits.
        
        Args:
            summary_text: The original summary text
            max_length: Maximum allowed length
            
        Returns:
            List of summary chunks that each fit within the limit
        """
        if len(summary_text) <= max_length:
            return [summary_text]
        
        # First try splitting by paragraphs (double newlines)
        chunks = []
        paragraphs = summary_text.split("\n\n")
        
        current_chunk = ""
        for paragraph in paragraphs:
            # If adding this paragraph would exceed the limit, start a new chunk
            if len(current_chunk) + len(paragraph) + 2 > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If a single paragraph is too long, split by sentences
                if len(paragraph) > max_length:
                    sentences = self._split_into_sentences(paragraph)
                    sentence_chunks = []
                    
                    current_sentence_chunk = ""
                    for sentence in sentences:
                        if len(current_sentence_chunk) + len(sentence) + 1 > max_length:
                            if current_sentence_chunk:
                                sentence_chunks.append(current_sentence_chunk)
                            
                            # If a single sentence is still too long, we have to split arbitrarily
                            if len(sentence) > max_length:
                                for i in range(0, len(sentence), max_length):
                                    sentence_chunks.append(sentence[i:i+max_length])
                            else:
                                current_sentence_chunk = sentence
                        else:
                            current_sentence_chunk += " " + sentence if current_sentence_chunk else sentence
                    
                    if current_sentence_chunk:
                        sentence_chunks.append(current_sentence_chunk)
                    
                    chunks.extend(sentence_chunks)
                else:
                    current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info(f"Split long summary into {len(chunks)} semantic chunks")
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Pattern for finding sentence boundaries
        sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
        sentences = sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_embedding_for_text(self, text: str) -> List[float]:
        """Generate embedding for a text chunk."""
        from services.ml.embeddings import EmbeddingService
        embedding_service = EmbeddingService()  # Or get from a service locator
        return embedding_service.generate_embedding(text)
    
    def supports_hybrid_search(self) -> bool:
        """
        Check if hybrid search is supported by the current configuration and collection.
        
        Returns:
            True if hybrid search is supported, False otherwise
        """
        if not self.hybrid_search_enabled:
            return False
        else:
            return True
        # Check if client supports hybrid search
        if hasattr(self.client, 'supports_hybrid_search'):
            return self.client.supports_hybrid_search()
        
        return False
    
    def _format_search_results(
        self, 
        results: List[VectorSearchResult], 
        is_hybrid: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Format search results into a standardized dictionary format.
        
        Args:
            results: List of VectorSearchResult objects
            is_hybrid: Whether these are results from hybrid search
            
        Returns:
            List of formatted result dictionaries
        """
        formatted_results = []
        
        for result in results:
            entity = result.entity
            
            # Create base result entry
            formatted_result = {
                "document_id": entity.document_id,
                "case_id": entity.case_id,
                "content": entity.content,
                "score": result.score,
                "metadata": entity.metadata or {},
                "content_type": entity.content_type,
                "chunk_type": entity.chunk_type,
                "page_number": entity.page_number,
                "tree_level": entity.tree_level,
                "search_method": "hybrid" if is_hybrid else "vector"
            }
            
            # Add original scores if available
            if is_hybrid and entity.metadata:
                if "vector_score" in entity.metadata:
                    formatted_result["vector_score"] = entity.metadata.get("vector_score")
                if "bm25_score" in entity.metadata:
                    formatted_result["bm25_score"] = entity.metadata.get("bm25_score")
                if "search_method" in entity.metadata:
                    formatted_result["search_method"] = entity.metadata.get("search_method")
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        case_ids: List[str],
        document_ids: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None,
        tree_levels: Optional[List[int]] = None,
        top_k: int = 5,
        vector_weight: float = 0.5,
        fusion_method: str = "weighted"
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and BM25 text matching.
        
        Args:
            query_text: Original query text for BM25 search
            query_embedding: Query embedding vector
            case_ids: List of case IDs to filter by
            document_ids: Optional list of document IDs to filter by
            content_types: Optional list of content types to filter by
            chunk_types: Optional list of chunk types to filter by
            tree_levels: Optional list of tree levels to filter by
            top_k: Number of results to return
            vector_weight: Weight for vector scores (0-1)
            fusion_method: Method for score fusion ("weighted" or "rrf")
            
        Returns:
            List of result dictionaries with content and metadata
        """
        if not self.supports_hybrid_search():
            logger.warning("Hybrid search not supported by this collection. Falling back to vector search.")
            return self.search(
                query_embedding=query_embedding,
                case_ids=case_ids,
                document_ids=document_ids,
                content_types=content_types,
                chunk_types=chunk_types,
                tree_levels=tree_levels,
                top_k=top_k
            )
        
        # Use the regular search method with hybrid enabled
        return self.search(
            query_embedding=query_embedding,
            case_ids=case_ids,
            document_ids=document_ids,
            content_types=content_types,
            chunk_types=chunk_types,
            tree_levels=tree_levels,
            top_k=top_k,
            use_hybrid=True,
            vector_weight=vector_weight,
            query_text=query_text,
            fusion_method=fusion_method
        )
        
    def text_search(
        self,
        query_text: str,
        case_ids: List[str],
        document_ids: Optional[List[str]] = None,
        content_types: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None,
        tree_levels: Optional[List[int]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform text-only search using BM25.
        
        Args:
            query_text: Query text for BM25 search
            case_ids: List of case IDs to filter by
            document_ids: Optional list of document IDs to filter by
            content_types: Optional list of content types to filter by
            chunk_types: Optional list of chunk types to filter by
            tree_levels: Optional list of tree levels to filter by
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with content and metadata
        """
        if not self.supports_hybrid_search():
            logger.warning("Text search requires BM25 support which is not available. Cannot proceed.")
            return []
        
        try:
            # Create base filter expression
            filter_parts = []
            
            # Add case_ids filter
            if case_ids:
                if len(case_ids) == 1:
                    filter_parts.append(f'case_id == "{case_ids[0]}"')
                else:
                    case_list = '", "'.join(case_ids)
                    filter_parts.append(f'case_id in ["{case_list}"]')
            
            # Add document_ids filter if provided
            if document_ids and len(document_ids) > 0:
                if len(document_ids) == 1:
                    filter_parts.append(f'document_id == "{document_ids[0]}"')
                else:
                    doc_list = '", "'.join(document_ids)
                    filter_parts.append(f'document_id in ["{doc_list}"]')
            
            # Add content_types filter if provided
            if content_types and len(content_types) > 0:
                if len(content_types) == 1:
                    filter_parts.append(f'content_type == "{content_types[0]}"')
                else:
                    type_list = '", "'.join(content_types)
                    filter_parts.append(f'content_type in ["{type_list}"]')
            
            # Add chunk_types filter if provided
            if chunk_types and len(chunk_types) > 0:
                if len(chunk_types) == 1:
                    filter_parts.append(f'chunk_type == "{chunk_types[0]}"')
                else:
                    type_list = '", "'.join(chunk_types)
                    filter_parts.append(f'chunk_type in ["{type_list}"]')
            
            # Add tree_levels filter if provided
            if tree_levels and len(tree_levels) > 0:
                if len(tree_levels) == 1:
                    filter_parts.append(f'tree_level == {tree_levels[0]}')
                else:
                    level_list = ', '.join(map(str, tree_levels))
                    filter_parts.append(f'tree_level in [{level_list}]')
            
            # Combine filter parts with AND
            filter_expr = " && ".join(filter_parts) if filter_parts else ""
            
            # Add text match expression
            if filter_expr:
                filter_expr += f" && TEXT_MATCH(content, '{query_text}')"
            else:
                filter_expr = f"TEXT_MATCH(content, '{query_text}')"
            
            # Execute text search using Milvus client directly
            # We need to bypass the regular search mechanism as it expects vector embeddings
            results = self.client.collection.search(
                data=[query_text],
                anns_field="sparse",  # Use sparse field for BM25
                param={"metric_type": "BM25", "params": {"k1": 1.5, "b": 0.75}},
                limit=top_k,
                expr=filter_expr,
                output_fields=["id", "document_id", "case_id", "chunk_id", "content", 
                            "content_type", "chunk_type", "page_number", "tree_level", 
                            "metadata"]
            )
            
            # Process results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    entity_id = str(hit.id)
                    document_id = self._extract_field(hit, "document_id")
                    case_id = self._extract_field(hit, "case_id", "default")
                    content = self._extract_field(hit, "content", "")
                    
                    formatted_result = {
                        "document_id": document_id,
                        "case_id": case_id,
                        "content": content,
                        "score": hit.score,
                        "metadata": self._extract_field(hit, "metadata", {}),
                        "content_type": self._extract_field(hit, "content_type", "text"),
                        "chunk_type": self._extract_field(hit, "chunk_type", "original"),
                        "page_number": self._extract_field(hit, "page_number"),
                        "tree_level": self._extract_field(hit, "tree_level", 0),
                        "search_method": "bm25"
                    }
                    
                    formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error performing text search: {str(e)}")
            return []
        
    def _extract_field(self, hit, field_name, default=None):
        """Extract a field from a hit result"""
        try:
            return getattr(hit.entity, field_name, default)
        except Exception:
            return default
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate results based on content or ID.
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated results preserving order
        """
        seen_ids = set()
        deduplicated = []
        
        for result in results:
            # Use combination of document_id and chunk_id as unique identifier
            unique_id = f"{result['document_id']}_{result.get('chunk_id', '')}"
            
            if unique_id not in seen_ids:
                seen_ids.add(unique_id)
                deduplicated.append(result)
        
        return deduplicated
        
    def get_collection_capabilities(self) -> Dict[str, bool]:
        """
        Get capabilities of the current collection.
        
        Returns:
            Dictionary of capabilities and their availability status
        """
        capabilities = {
            "vector_search": True,  # Always available
            "hybrid_search": self.supports_hybrid_search(),
            "bm25_search": self.supports_hybrid_search(),  # Same as hybrid for now
        }
        
        return capabilities