import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
import uuid

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema as MilvusCollectionSchema,
    DataType,
    Collection,
)

from .schemas import (
    CollectionSchema,
    ScalarField,
    VectorField,
    VectorEntity,
    VectorSearchParams,
    VectorSearchResult
)

logger = logging.getLogger(__name__)


class MilvusClient:
    """
    Enhanced Milvus client with support for partitioning and explicit fields.
    This is an upgrade from the original MilvusVectorStore in rag_system.py.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        user: str = "",
        password: str = "",
        collection_name: str = "document_store",
        dimension: int = 768,
        schema: Optional[CollectionSchema] = None,
        hybrid_search_enabled: bool = True  # New parameter
    ):
        """
        Initialize the Milvus client.
        
        Args:
            host: Milvus server host
            port: Milvus server port
            user: Milvus username (if authentication is enabled)
            password: Milvus password (if authentication is enabled)
            collection_name: Name of the collection to use
            dimension: Dimension of the embedding vectors
            schema: Custom schema (if None, a default schema will be created)
            hybrid_search_enabled: Whether to enable hybrid search support
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.collection_name = collection_name
        self.dimension = dimension
        self.hybrid_search_enabled = hybrid_search_enabled
        
        # Use provided schema or create a default one
        self.schema = schema or CollectionSchema.default_document_schema(
            collection_name=collection_name,
            dimension=dimension
        )
        
        # Disable sparse field if hybrid search is not enabled
        if not hybrid_search_enabled and hasattr(self.schema, 'sparse_field'):
            self.schema.sparse_field = None
            self.schema.bm25_function = None
        
        # Collection instance (initialized later)
        self.collection = None
        
        # Track created partitions to avoid duplication errors
        self.created_partitions = set()
        
        # Connect to Milvus
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Milvus server"""
        try:
            # Connect to Milvus
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            logger.info(f"Connected to Milvus server at {self.host}:{self.port}")
            
            # Initialize collection if it doesn't exist
            if not utility.has_collection(self.collection_name):
                self._create_collection()
            else:
                self._load_collection()
        except Exception as e:
            logger.error(f"Error connecting to Milvus: {str(e)}")
            raise
    
    def _data_type_to_milvus(self, field_type: str) -> DataType:
        """Convert string data type to Milvus DataType"""
        type_mapping = {
            "VARCHAR": DataType.VARCHAR,
            "INT64": DataType.INT64,
            "FLOAT": DataType.DOUBLE,
            "BOOL": DataType.BOOL,
            "JSON": DataType.JSON
        }
        
        return type_mapping.get(field_type, DataType.VARCHAR)
    
    def _create_collection(self) -> None:
        """Create a new collection using the schema with BM25 support"""
        try:
            logger.info(f"Creating collection {self.collection_name}")
            
            # Convert schema fields to Milvus FieldSchema objects
            fields = []
            
            # Add scalar fields
            for field in self.schema.scalar_fields:
                field_args = {
                    "name": field.name,
                    "dtype": self._data_type_to_milvus(field.field_type),
                    "is_primary": field.is_primary
                }
                
                # Add max_length for VARCHAR fields
                if field.field_type == "VARCHAR" and field.max_length:
                    field_args["max_length"] = field.max_length
                
                # Add analyzer support for content field
                if field.name == "content" or getattr(field, "enable_analyzer", False):
                    field_args["enable_analyzer"] = True
                    
                    # If we want to use exact text matching like "foo" == "foo", add this
                    if getattr(field, "enable_match", False):
                        field_args["enable_match"] = True
                
                fields.append(FieldSchema(**field_args))
            
            # Add vector field
            vector_field = self.schema.vector_field
            fields.append(FieldSchema(
                name=vector_field.name,
                dtype=DataType.FLOAT_VECTOR,
                dim=vector_field.dimension
            ))
            
            # Add sparse vector field for BM25 if defined in schema
            if self.schema.sparse_field:
                sparse_field = self.schema.sparse_field
                fields.append(FieldSchema(
                    name=sparse_field.name,
                    dtype=DataType.SPARSE_FLOAT_VECTOR
                ))
            
            # Create Milvus schema
            milvus_schema = MilvusCollectionSchema(
                fields=fields, 
                description=self.schema.description
            )
            
            # Add BM25 function if defined in schema
            if self.schema.bm25_function:
                bm25_config = self.schema.bm25_function
                bm25_function = Function(
                    name=f"{bm25_config.input_field}_bm25_emb",
                    input_field_names=[bm25_config.input_field],
                    output_field_names=[bm25_config.output_field],
                    function_type=FunctionType.BM25
                )
                milvus_schema.add_function(bm25_function)
            
            # Create collection
            self.collection = Collection(
                name=self.collection_name,
                schema=milvus_schema,
                shards_num=2  # Using multiple shards for better performance
            )
            
            # Create index for vector field
            index_params = {
                "index_type": vector_field.index_type,
                "metric_type": vector_field.metric_type,
                "params": vector_field.params
            }
            
            self.collection.create_index(field_name=vector_field.name, index_params=index_params)
            logger.info(f"Created index on {vector_field.name} field")
            
            # Create sparse index for BM25 search if sparse field is defined
            if self.schema.sparse_field:
                sparse_field = self.schema.sparse_field
                sparse_index_params = {
                    "index_type": sparse_field.index_type,
                    "metric_type": sparse_field.metric_type,
                    "params": sparse_field.params
                }
                
                self.collection.create_index(
                    field_name=sparse_field.name, 
                    index_params=sparse_index_params
                )
                logger.info(f"Created sparse index on {sparse_field.name} field")
            
            # Load collection
            self.collection.load()
            logger.info(f"Collection {self.collection_name} created and loaded")
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    def _load_collection(self) -> None:
        """Load an existing collection"""
        try:
            logger.info(f"Loading collection {self.collection_name}")
            self.collection = Collection(name=self.collection_name)
            self.collection.load()
            logger.info(f"Collection {self.collection_name} loaded")
        except Exception as e:
            logger.error(f"Error loading collection: {str(e)}")
            raise
    
    def _ensure_partition_exists(self, partition_key: str) -> str:
        """
        Ensure a partition exists, creating it if necessary.
        
        Args:
            partition_key: Base key for the partition
            
        Returns:
            The partition name
        """
        try:
            # Check partition name length (Milvus has a limit)
            partition_name = f"p_{partition_key}"
            
            if len(partition_name) > 50:
                # Create a hash of the partition key to shorten it
                hash_key = hashlib.md5(partition_key.encode()).hexdigest()[:10]
                partition_name = f"p_{hash_key}"
            
            # Only create if not in our tracking set
            if partition_name not in self.created_partitions:
                # Get existing partitions
                existing_partitions = [p.name for p in self.collection.partitions]
                
                if partition_name not in existing_partitions:
                    # Create partition
                    self.collection.create_partition(partition_name)
                    logger.info(f"Created partition {partition_name}")
                
                # Add to tracking set regardless
                self.created_partitions.add(partition_name)
            
            return partition_name
                
        except Exception as e:
            logger.error(f"Error ensuring partition exists: {e}")
            # Continue without partitioning in case of error
            logger.warning("Continuing without partitioning")
            return ""
    
    def add_entities(self, entities: List[VectorEntity], batch_size: int = 100) -> bool:
        """
        Add entities to the collection.
        
        Args:
            entities: List of VectorEntity objects to add
            batch_size: Number of entities to insert in one batch
            
        Returns:
            True if successful, False otherwise
        """
        if not entities:
            logger.warning("No entities provided for insertion")
            return False
        
        try:
            # Load collection if not already loaded
            if not self.collection:
                self._load_collection()
            
            # First pass: create all needed partitions to avoid errors during insertion
            doc_partitions = {}
            for entity in entities:
                doc_id = entity.document_id
                if doc_id not in doc_partitions:
                    partition_name = self._ensure_partition_exists(doc_id)
                    doc_partitions[doc_id] = partition_name
            
            # Process in batches
            total_inserted = 0
            total_entities = len(entities)
            
            # Debug: Check the first entity's embedding
            if entities and hasattr(entities[0], 'embedding'):
                first_emb = entities[0].embedding
                if first_emb:
                    logger.info(f"First entity embedding type: {type(first_emb)}, dimension: {len(first_emb) if isinstance(first_emb, list) else 'not a list'}")
            
            for i in range(0, total_entities, batch_size):
                # Get current batch
                batch = entities[i:i+batch_size]
                
                # Normalize embeddings - ensure all have correct dimension
                valid_batch = []
                for entity in batch:
                    # Make a copy to avoid modifying the original
                    processed_entity = self._process_entity_embedding(entity)
                    valid_batch.append(processed_entity)
                
                if not valid_batch:
                    logger.warning(f"No valid entities in batch {i//batch_size + 1}")
                    continue
                    
                # Convert entities to Milvus format
                ids = []
                document_ids = []
                case_ids = []
                chunk_ids = []
                contents = []
                content_types = []
                chunk_types = []
                page_numbers = []
                tree_levels = []
                metadatas = []
                embeddings = []
                
                for entity in valid_batch:
                    # Extract fields
                    ids.append(entity.id)
                    document_ids.append(entity.document_id)
                    case_ids.append(entity.case_id)
                    chunk_ids.append(entity.chunk_id)
                    contents.append(entity.content)
                    content_types.append(entity.content_type)
                    chunk_types.append(entity.chunk_type)
                    page_numbers.append(entity.page_number if entity.page_number is not None else -1)
                    tree_levels.append(entity.tree_level if entity.tree_level is not None else 0)
                    metadatas.append(entity.metadata)
                    embeddings.append(entity.embedding)
                
                # Verify all lists have the same length
                list_lengths = {
                    "ids": len(ids),
                    "document_ids": len(document_ids),
                    "case_ids": len(case_ids),
                    "chunk_ids": len(chunk_ids),
                    "contents": len(contents),
                    "content_types": len(content_types),
                    "chunk_types": len(chunk_types),
                    "page_numbers": len(page_numbers),
                    "tree_levels": len(tree_levels),
                    "metadatas": len(metadatas),
                    "embeddings": len(embeddings)
                }
                
                # Log lengths for debugging
                logger.debug(f"List lengths for batch insertion: {list_lengths}")
                
                if len(set(list_lengths.values())) != 1:
                    logger.error(f"Inconsistent list lengths for batch insertion: {list_lengths}")
                    continue
                
                # Insert data
                insert_data = [
                    ids,
                    document_ids,
                    case_ids,
                    chunk_ids,
                    contents,
                    content_types,
                    chunk_types,
                    page_numbers,
                    tree_levels,
                    metadatas,
                    embeddings
                ]
                
                # Determine which partition to use for this batch
                # If mixed document IDs, don't specify partition
                unique_doc_ids = set(document_ids)
                insert_partition = None
                if len(unique_doc_ids) == 1:
                    doc_id = next(iter(unique_doc_ids))
                    insert_partition = doc_partitions.get(doc_id)
                
                # Perform insertion
                try:
                    self.collection.insert(
                        data=insert_data,
                        partition_name=insert_partition
                    )
                    total_inserted += len(valid_batch)
                    logger.info(f"Inserted batch of {len(valid_batch)} entities ({total_inserted}/{total_entities})")
                except Exception as batch_error:
                    logger.error(f"Error inserting batch: {batch_error}")
                    # Log the first entity's embedding information for debugging
                    if embeddings and len(embeddings) > 0:
                        logger.debug(f"First embedding length: {len(embeddings[0])}")
                        logger.debug(f"Embedding dimension should be: {self.dimension}")
                        if isinstance(embeddings[0], list) and len(embeddings[0]) > 0:
                            logger.debug(f"First embedding type: {type(embeddings[0][0])}")
            
            # Flush to ensure data is persisted
            if total_inserted > 0:
                self.collection.flush()
                logger.info(f"Successfully inserted {total_inserted} entities")
            
            return total_inserted > 0
        except Exception as e:
            logger.error(f"Error adding entities: {e}")
            return False

    def _process_entity_embedding(self, entity: VectorEntity) -> VectorEntity:
        """
        Process an entity's embedding to ensure it has the correct dimension.
        Creates a new entity to avoid modifying the original.
        
        Args:
            entity: The VectorEntity to process
            
        Returns:
            A new VectorEntity with processed embedding
        """
        # Copy entity to avoid modifying the original
        import copy
        processed = copy.deepcopy(entity)
        
        # Handle embedding issues
        if not processed.embedding or not isinstance(processed.embedding, list):
            logger.warning(f"Entity {processed.id} has invalid embedding type, creating zero vector")
            processed.embedding = [0.0] * self.dimension
            return processed
        
        # Check embedding dimension
        emb_dim = len(processed.embedding)
        if emb_dim != self.dimension:
            logger.warning(f"Entity {processed.id} has dimension mismatch: {emb_dim} != {self.dimension}")
            
            # Fix dimension by padding or truncating
            if emb_dim < self.dimension:
                # Pad with zeros
                processed.embedding = processed.embedding + [0.0] * (self.dimension - emb_dim)
                logger.info(f"Padded embedding for {processed.id} from {emb_dim} to {self.dimension}")
            else:
                # Truncate
                processed.embedding = processed.embedding[:self.dimension]
                logger.info(f"Truncated embedding for {processed.id} from {emb_dim} to {self.dimension}")
        
        return processed
    
    def _extract_hit_field(self, hit, field_name, default=None):
        """
        Extract a field from a hit entity compatible with your Milvus version.
        
        Args:
            hit: The Milvus Hit object
            field_name: The field name to extract
            default: Default value if the field is not found
            
        Returns:
            The field value or default
        """
        try:
            # Based on your working example, direct attribute access is likely to work
            return getattr(hit.entity, field_name, default)
        except Exception as e:
            logger.warning(f"Error extracting field {field_name} from hit: {e}")
            return default
    
    def search(
        self, 
        search_params: VectorSearchParams,
        use_hybrid: bool = True,
        query_text: Optional[str] = None,
        vector_weight: float = 0.5
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors with optional hybrid BM25 search.
        
        Args:
            search_params: Parameters for the search
            use_hybrid: Whether to use hybrid search (vector + BM25)
            query_text: Original query text for BM25 (required if use_hybrid=True)
            vector_weight: Weight for vector scores in hybrid search (0-1)
            
        Returns:
            List of search results
        """
        try:
            # Determine if we can and should use hybrid search
            can_use_hybrid = self.hybrid_search_enabled and self.supports_hybrid_search()
            should_use_hybrid = use_hybrid and can_use_hybrid and query_text
            
            if use_hybrid and not can_use_hybrid:
                logger.warning("Hybrid search requested but not supported by collection. Falling back to vector search.")
                
            if use_hybrid and not query_text:
                logger.warning("Hybrid search requested but no query_text provided. Falling back to vector search.")
                
            # Use hybrid search if requested and supported
            if should_use_hybrid:
                return self.hybrid_search(
                    query_embedding=search_params.query_embedding,
                    query_text=query_text,
                    search_params=search_params,
                    vector_weight=vector_weight
                )
            
            # Otherwise, use regular vector search (existing implementation)
            # Load collection if not already loaded
            if not self.collection:
                self._load_collection()
            
            # Generate filter expression
            expr = search_params.get_filter_expr()
            
            # Prepare search parameters
            vector_field = self.schema.vector_field.name
            search_params_dict = {
                "metric_type": search_params.metric_type,
                "params": search_params.params
            }
            
            # Define output fields
            output_fields = ["document_id", "chunk_id", "content", "content_type", 
                        "chunk_type", "page_number", "tree_level", "metadata"]
            
            # Execute search
            start_time = time.time()
            results = self.collection.search(
                data=[search_params.query_embedding],
                anns_field=vector_field,
                param=search_params_dict,
                limit=search_params.top_k,
                expr=expr,
                output_fields=output_fields
            )
            search_time = time.time() - start_time
            
            # Process results
            search_results = []
            
            for hits in results:
                for hit in hits:
                    # Create entity with simpler field access
                    entity = VectorEntity(
                        id=str(hit.id),
                        document_id=self._extract_hit_field(hit, "document_id", ""),
                        chunk_id=self._extract_hit_field(hit, "chunk_id", ""),
                        content=self._extract_hit_field(hit, "content", ""),
                        content_type=self._extract_hit_field(hit, "content_type", "text"),
                        chunk_type=self._extract_hit_field(hit, "chunk_type", "original"),
                        page_number=self._extract_hit_field(hit, "page_number"),
                        tree_level=self._extract_hit_field(hit, "tree_level", 0),
                        metadata=self._extract_hit_field(hit, "metadata", {}),
                        embedding=[]  # Embedding not returned in search results
                    )
                    
                    # Add search method metadata
                    entity.metadata["search_method"] = "vector"
                    
                    # Create search result
                    search_result = VectorSearchResult(
                        entity=entity,
                        score=hit.score,
                        distance=0.0  # May not be available in all Milvus versions
                    )
                    
                    search_results.append(search_result)
            
            logger.info(f"Search completed in {search_time:.3f}s with {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def hybrid_search(
        self, 
        query_embedding: List[float],
        query_text: str,
        search_params: VectorSearchParams,
        vector_weight: float = 0.5
    ) -> List[VectorSearchResult]:
        """
        Perform hybrid search combining vector similarity with BM25 text matching.
        
        Args:
            query_embedding: Query embedding vector
            query_text: Original query text for BM25 search
            search_params: Parameters for the search
            vector_weight: Weight for vector scores (0-1), BM25 weight will be 1-vector_weight
            
        Returns:
            List of search results
        """
        try:
            # Load collection if not already loaded
            if not self.collection:
                self._load_collection()
            
            # 1. First, execute vector search
            logger.info(f"Executing vector search with embedding dimension {len(query_embedding)}")
            vector_results = self.search(search_params)
            
            # 2. Prepare BM25 search parameters
            vector_field = self.schema.vector_field.name
            sparse_field = self.schema.sparse_field.name if self.schema.sparse_field else "sparse"
            
            # Get filter expression from search params
            expr = search_params.get_filter_expr()
            
            # Modify the expression to include TEXT_MATCH for BM25
            bm25_expr = expr
            if bm25_expr:
                bm25_expr += f" && TEXT_MATCH(content, '{query_text}')"
            else:
                bm25_expr = f"TEXT_MATCH(content, '{query_text}')"
            
            # BM25 search parameters
            bm25_params = {
                "metric_type": "BM25",
                "params": {"k1": 1.5, "b": 0.75}  # BM25 parameters
            }
            
            # Define output fields
            output_fields = ["document_id", "chunk_id", "content", "content_type", 
                        "chunk_type", "page_number", "tree_level", "metadata"]
            
            # 3. Execute BM25 search
            logger.info(f"Executing BM25 search with query: {query_text}")
            bm25_results = self.collection.search(
                data=[query_text],
                anns_field=sparse_field,
                param=bm25_params,
                limit=search_params.top_k * 2,  # Get more results for better fusion
                expr=bm25_expr,
                output_fields=output_fields
            )
            
            # 4. Process BM25 results to match vector results format
            bm25_search_results = []
            
            for hits in bm25_results:
                for hit in hits:
                    # Create entity with extracted fields
                    entity = VectorEntity(
                        id=str(hit.id),
                        document_id=self._extract_hit_field(hit, "document_id", ""),
                        chunk_id=self._extract_hit_field(hit, "chunk_id", ""),
                        content=self._extract_hit_field(hit, "content", ""),
                        content_type=self._extract_hit_field(hit, "content_type", "text"),
                        chunk_type=self._extract_hit_field(hit, "chunk_type", "original"),
                        page_number=self._extract_hit_field(hit, "page_number"),
                        tree_level=self._extract_hit_field(hit, "tree_level", 0),
                        metadata=self._extract_hit_field(hit, "metadata", {}),
                        embedding=[]  # Embedding not returned in search results
                    )
                    
                    # Create search result
                    search_result = VectorSearchResult(
                        entity=entity,
                        score=hit.score,
                        distance=0.0
                    )
                    
                    bm25_search_results.append(search_result)
            
            # 5. Fuse vector and BM25 results
            combined_results = self._fuse_search_results(
                vector_results, 
                bm25_search_results, 
                vector_weight
            )
            
            logger.info(f"Hybrid search completed with {len(combined_results)} combined results")
            return combined_results[:search_params.top_k]  # Return top-k results
        
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fall back to vector search
            logger.info("Falling back to vector-only search")
            return self.search(search_params)
    
    def delete_by_document_ids(self, document_ids: List[str]) -> bool:
        """
        Delete entities by document IDs.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load collection if not already loaded
            if not self.collection:
                self._load_collection()
            
            # Build expression to delete by document_id
            if len(document_ids) == 1:
                expr = f'document_id == "{document_ids[0]}"'
            else:
                docs_str = '", "'.join(document_ids)
                expr = f'document_id in ["{docs_str}"]'
            
            # Execute deletion
            self.collection.delete(expr)
            
            # Flush to ensure changes are committed
            self.collection.flush()
            
            logger.info(f"Deleted entities for document IDs: {document_ids}")
            
            return True
        except Exception as e:
            logger.error(f"Error deleting by document IDs: {e}")
            return False
    
    def delete_by_filter(self, filter_expr: str) -> bool:
        """
        Delete entities by filter expression.
        
        Args:
            filter_expr: Milvus filter expression
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load collection if not already loaded
            if not self.collection:
                self._load_collection()
            
            # Execute deletion
            self.collection.delete(filter_expr)
            
            # Flush to ensure changes are committed
            self.collection.flush()
            
            logger.info(f"Deleted entities with filter: {filter_expr}")
            
            return True
        except Exception as e:
            logger.error(f"Error deleting by filter: {str(e)}")
            return False
    
    def count_entities(self, filter_expr: Optional[str] = None) -> int:
        """
        Count entities in the collection.
        
        Args:
            filter_expr: Optional filter expression
            
        Returns:
            Number of entities
        """
        try:
            # Load collection if not already loaded
            if not self.collection:
                self._load_collection()
            
            # Count entities
            if filter_expr:
                # For filtered counts, we need to use the query method
                try:
                    results = self.collection.query(
                        expr=filter_expr,
                        output_fields=["count(*)"]
                    )
                    if results and len(results) > 0:
                        return results[0].get("count(*)", 0)
                    return 0
                except Exception as query_error:
                    logger.warning(f"Error using query for count: {query_error}")
                    # Fall back to search count
                    search_params = {
                        "metric_type": "L2",
                        "params": {"nprobe": 10}
                    }
                    results = self.collection.search(
                        data=[[0.0] * self.dimension],  # Dummy vector
                        anns_field=self.schema.vector_field.name,
                        param=search_params,
                        limit=10000,  # Very high limit to count
                        expr=filter_expr,
                        output_fields=["count(*)"]
                    )
                    return len(results[0]) if results else 0
            else:
                # For total count, use collection.num_entities
                return self.collection.num_entities
        except Exception as e:
            logger.error(f"Error counting entities: {e}")
            return 0
    
    def release(self) -> None:
        """Release collection from memory"""
        if self.collection:
            try:
                self.collection.release()
                logger.info(f"Released collection {self.collection_name}")
            except Exception as e:
                logger.error(f"Error releasing collection: {str(e)}")
    
    def drop(self) -> None:
        """Drop the collection"""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped collection {self.collection_name}")
                self.collection = None
        except Exception as e:
            logger.error(f"Error dropping collection: {str(e)}")
    
    def get_partition_statistics(self) -> Dict[str, int]:
        """
        Get statistics about partitions.
        
        Returns:
            Dictionary of partition names and entity counts
        """
        try:
            # Load collection if not already loaded
            if not self.collection:
                self._load_collection()
            
            # Get partitions
            partitions = self.collection.partitions
            
            # Count entities in each partition
            stats = {}
            for partition in partitions:
                partition_name = partition.name
                try:
                    # In newer Milvus versions, you can use this info directly
                    stats[partition_name] = partition.num_entities
                except AttributeError:
                    # For older versions, fall back to querying
                    try:
                        # Try direct query first
                        results = self.collection.query(
                            expr="",
                            output_fields=["count(*)"],
                            partition_names=[partition_name]
                        )
                        if results and len(results) > 0:
                            stats[partition_name] = results[0].get("count(*)", 0)
                        else:
                            stats[partition_name] = 0
                    except Exception:
                        # Last resort - count all entities in the partition
                        stats[partition_name] = 0
            
            return stats
        except Exception as e:
            logger.error(f"Error getting partition statistics: {e}")
            return {}
        
    def _fuse_search_results(
        self,
        vector_results: List[VectorSearchResult],
        bm25_results: List[VectorSearchResult],
        vector_weight: float = 0.5
    ) -> List[VectorSearchResult]:
        """
        Fuse vector and BM25 search results using min-max normalization and weighted combination.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            vector_weight: Weight for vector scores (0-1)
            
        Returns:
            Combined and re-ranked results
        """
        # Check if either result set is empty
        if not vector_results:
            return bm25_results
        if not bm25_results:
            return vector_results
        
        # 1. Extract scores for normalization
        vector_scores = [result.score for result in vector_results]
        bm25_scores = [result.score for result in bm25_results]
        
        # Handle edge cases where all scores are the same
        v_min, v_max = min(vector_scores), max(vector_scores)
        if v_min == v_max:
            v_min, v_max = 0.0, 1.0  # Avoid division by zero
            
        b_min, b_max = min(bm25_scores), max(bm25_scores)
        if b_min == b_max:
            b_min, b_max = 0.0, 1.0  # Avoid division by zero
        
        # 2. Define normalization function
        def normalize(score, min_score, max_score):
            if max_score == min_score:
                return 0.5  # Default to middle value if all scores are the same
            return (score - min_score) / (max_score - min_score)
        
        # 3. Build a merged map keyed by entity ID to handle duplicates
        merged = {}
        
        # Process vector results
        for result in vector_results:
            entity_id = result.entity.id
            merged[entity_id] = {
                "entity": result.entity,
                "vector_score": normalize(result.score, v_min, v_max),
                "bm25_score": 0.0,  # Default BM25 score if not found
                "used_source": "vector"
            }
        
        # Process BM25 results
        for result in bm25_results:
            entity_id = result.entity.id
            if entity_id in merged:
                # Entity already exists from vector search, just add BM25 score
                merged[entity_id]["bm25_score"] = normalize(result.score, b_min, b_max)
                merged[entity_id]["used_source"] = "both"
            else:
                # New entity from BM25 search
                merged[entity_id] = {
                    "entity": result.entity,
                    "vector_score": 0.0,  # Default vector score if not found
                    "bm25_score": normalize(result.score, b_min, b_max),
                    "used_source": "bm25"
                }
        
        # 4. Combine scores using weighted sum
        combined_results = []
        for entity_data in merged.values():
            combined_score = (
                vector_weight * entity_data["vector_score"] + 
                (1 - vector_weight) * entity_data["bm25_score"]
            )
            
            # Add metadata about score sources
            entity_data["entity"].metadata["search_method"] = entity_data["used_source"]
            entity_data["entity"].metadata["vector_score"] = entity_data["vector_score"]
            entity_data["entity"].metadata["bm25_score"] = entity_data["bm25_score"]
            
            # Create result with combined score
            combined_results.append(
                VectorSearchResult(
                    entity=entity_data["entity"],
                    score=combined_score,
                    distance=0.0  # Set distance to 0 since it's not relevant for combined score
                )
            )
        
        # 5. Sort by combined score (highest first) and return
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results
    
    def supports_hybrid_search(self) -> bool:
        """
        Check if the current collection supports hybrid search.
        
        Returns:
            True if hybrid search is supported, False otherwise
        """
        try:
            if not self.collection:
                self._load_collection()
                
            # Check if sparse field exists
            fields = self.collection.schema.fields
            has_sparse = any(field.name == "sparse" and field.dtype == DataType.SPARSE_FLOAT_VECTOR 
                            for field in fields)
            
            # Check if content field has analyzer enabled
            has_analyzer = any(field.name == "content" and hasattr(field, "enable_analyzer") and 
                            field.enable_analyzer for field in fields)
            
            return has_sparse and has_analyzer
            
        except Exception as e:
            logger.warning(f"Error checking hybrid search support: {str(e)}")
            return False
        
    def _fuse_with_rrf(
        self,
        vector_results: List[VectorSearchResult],
        bm25_results: List[VectorSearchResult],
        k: int = 60  # Constant in RRF algorithm
    ) -> List[VectorSearchResult]:
        """
        Fuse results using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum(1 / (k + rank_i)) where rank_i is the rank of the document in result list i.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            k: Constant that controls the impact of lower-ranked documents
            
        Returns:
            Combined and re-ranked results
        """
        # Build maps of entity ID to rank for each result set
        vector_ranks = {result.entity.id: i+1 for i, result in enumerate(vector_results)}
        bm25_ranks = {result.entity.id: i+1 for i, result in enumerate(bm25_results)}
        
        # Collect all unique entity IDs
        all_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        
        # Compute RRF score for each entity
        rrf_scores = {}
        merged_entities = {}
        
        for entity_id in all_ids:
            # Get ranks (use a large value if not present in a result set)
            v_rank = vector_ranks.get(entity_id, len(vector_ranks) + 1000)
            b_rank = bm25_ranks.get(entity_id, len(bm25_ranks) + 1000)
            
            # Compute RRF score
            rrf_score = 1/(k + v_rank) + 1/(k + b_rank)
            rrf_scores[entity_id] = rrf_score
            
            # Store entity
            if entity_id in vector_ranks:
                for result in vector_results:
                    if result.entity.id == entity_id:
                        merged_entities[entity_id] = result.entity
                        break
            else:
                for result in bm25_results:
                    if result.entity.id == entity_id:
                        merged_entities[entity_id] = result.entity
                        break
        
        # Create combined results
        combined_results = []
        for entity_id, score in rrf_scores.items():
            entity = merged_entities.get(entity_id)
            if entity:
                # Add search method metadata
                entity.metadata["search_method"] = "hybrid_rrf"
                combined_results.append(
                    VectorSearchResult(
                        entity=entity,
                        score=score,
                        distance=0.0
                    )
                )
        
        # Sort by RRF score (highest first) and return
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results


# Helper function to create a vector entity from document chunk
def create_vector_entity(
    document_id: str,
    chunk_id: str,
    content: str,
    embedding: List[float],
    case_id: str = "default",
    content_type: str = "text",
    chunk_type: str = "original",
    page_number: Optional[int] = None,
    tree_level: int = 0,
    metadata: Optional[Dict[str, Any]] = None
) -> VectorEntity:
    """
    Create a VectorEntity from document chunk information.
    
    Args:
        document_id: Document ID
        chunk_id: Chunk ID
        content: Text content
        embedding: Vector embedding
        case_id: Case ID for grouping documents
        content_type: Type of content (text, table, image, etc.)
        chunk_type: Type of chunk (original, summary, etc.)
        page_number: Page number in the original document
        tree_level: RAPTOR tree level (0 for original chunks)
        metadata: Additional metadata
        
    Returns:
        VectorEntity instance
    """
    # Generate a unique ID based on document_id and chunk_id
    entity_id = f"{document_id}_{chunk_id}"
    
    # Create entity
    entity = VectorEntity(
        id=entity_id,
        document_id=document_id,
        case_id=case_id,
        chunk_id=chunk_id,
        content=content,
        embedding=embedding,
        content_type=content_type,
        chunk_type=chunk_type,
        page_number=page_number,
        tree_level=tree_level,
        metadata=metadata or {}
    )
    
    return entity