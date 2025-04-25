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
        schema: Optional[CollectionSchema] = None
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
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.collection_name = collection_name
        self.dimension = dimension
        
        # Use provided schema or create a default one
        self.schema = schema or CollectionSchema.default_document_schema(
            collection_name=collection_name,
            dimension=dimension
        )
        
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
        """Create a new collection using the schema"""
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
                
                fields.append(FieldSchema(**field_args))
            
            # Add vector field
            vector_field = self.schema.vector_field
            fields.append(FieldSchema(
                name=vector_field.name,
                dtype=DataType.FLOAT_VECTOR,
                dim=vector_field.dimension
            ))
            
            # Create Milvus schema
            milvus_schema = MilvusCollectionSchema(fields=fields, description=self.schema.description)
            
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
            
            for i in range(0, total_entities, batch_size):
                # Get current batch
                batch = entities[i:i+batch_size]
                
                # Convert entities to Milvus format
                ids = []
                document_ids = []
                chunk_ids = []
                contents = []
                content_types = []
                chunk_types = []
                page_numbers = []
                tree_levels = []
                metadatas = []
                embeddings = []
                
                for entity in batch:
                    # Extract fields
                    ids.append(entity.id)
                    document_ids.append(entity.document_id)
                    chunk_ids.append(entity.chunk_id)
                    contents.append(entity.content)
                    content_types.append(entity.content_type)
                    chunk_types.append(entity.chunk_type)
                    page_numbers.append(entity.page_number if entity.page_number is not None else -1)
                    tree_levels.append(entity.tree_level if entity.tree_level is not None else 0)
                    metadatas.append(entity.metadata)
                    embeddings.append(entity.embedding)
                
                # Insert data
                insert_data = [
                    ids,
                    document_ids,
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
                self.collection.insert(
                    data=insert_data,
                    partition_name=insert_partition
                )
                total_inserted += len(batch)
                
                logger.info(f"Inserted batch of {len(batch)} entities ({total_inserted}/{total_entities})")
            
            # Flush to ensure data is persisted
            self.collection.flush()
            logger.info(f"Successfully inserted {total_inserted} entities")
            
            return True
        except Exception as e:
            logger.error(f"Error adding entities: {e}")
            return False
    
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
    
    def search(self, search_params: VectorSearchParams) -> List[VectorSearchResult]:
        """
        Search for similar vectors.
        
        Args:
            search_params: Parameters for the search
            
        Returns:
            List of search results
        """
        try:
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