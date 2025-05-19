import logging
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VectorField(BaseModel):
    """Definition of a vector field in Milvus"""
    name: str
    dimension: int
    metric_type: str = "COSINE"  # "COSINE", "L2", "IP"
    index_type: str = "HNSW"  # "HNSW", "FLAT", etc.
    params: Dict[str, Any] = Field(
        default_factory=lambda: {"M": 16, "efConstruction": 128}
    )


class ScalarField(BaseModel):
    """Definition of a scalar field in Milvus"""
    name: str
    field_type: str  # "VARCHAR", "INT64", "FLOAT", "BOOL", "JSON"
    is_primary: bool = False
    max_length: Optional[int] = None  # For VARCHAR type
    description: Optional[str] = None
    enable_analyzer: bool = False  # Add this field for text analysis
    enable_match: bool = False 

class SparseField(BaseModel):
    """Definition of a sparse vector field for BM25 search in Milvus"""
    name: str
    index_type: str = "SPARSE_INVERTED_INDEX"  # Default to sparse inverted index
    metric_type: str = "BM25"  # Default to BM25 for text search
    params: Dict[str, Any] = Field(
        default_factory=lambda: {"k1": 1.5, "b": 0.75}  # Default BM25 parameters
    )

class PartitionConfig(BaseModel):
    """Configuration for Milvus partitioning"""
    strategy: str = "case_id"  # "document_id", "case_id", "hybrid"
    field: str = "case_id"
    
    def get_partition_key(self, entity_data: Dict[str, Any]) -> str:
        """Generate partition key based on entity data and strategy"""
        if self.strategy == "document_id":
            return f"doc_{entity_data.get('document_id', 'unknown')}"
        elif self.strategy == "case_id":
            return f"case_{entity_data.get('case_id', 'unknown')}"
        elif self.strategy == "hybrid":
            doc_id = entity_data.get('document_id', 'unknown')
            case_id = entity_data.get('case_id', 'unknown')
            return f"case_{case_id}_doc_{doc_id}"
        else:
            return "default"

class BM25Function(BaseModel):
    """Configuration for BM25 function in Milvus"""
    input_field: str = "content"  # Field containing text
    output_field: str = "sparse"  # Field to store sparse vectors
    params: Dict[str, Any] = Field(
        default_factory=lambda: {"k1": 1.5, "b": 0.75}
    )

class CollectionSchema(BaseModel):
    """Schema definition for a Milvus collection"""
    name: str
    description: Optional[str] = None
    scalar_fields: List[ScalarField]
    vector_field: VectorField
    sparse_field: Optional[SparseField] = None  # Add optional sparse field
    bm25_function: Optional[BM25Function] = None 
    partition_config: PartitionConfig
    
    @classmethod
    def default_document_schema(cls, collection_name: str = "document_store", dimension: int = 3072) -> "CollectionSchema":
        """Create a default schema for document storage with BM25 support"""
        scalar_fields = [
            ScalarField(name="id", field_type="VARCHAR", is_primary=True, max_length=100, 
                    description="Unique entity identifier"),
            ScalarField(name="document_id", field_type="VARCHAR", max_length=100, 
                    description="Document identifier"),
            ScalarField(name="case_id", field_type="VARCHAR", max_length=256, 
                    description="Case identifier for grouping documents"),
            ScalarField(name="chunk_id", field_type="VARCHAR", max_length=100, 
                    description="Chunk identifier within document"),
            ScalarField(name="content", field_type="VARCHAR", max_length=65535, 
                    description="Text content of the chunk",
                    enable_analyzer=True, enable_match=True),  # Enable analyzer and match for BM25
            ScalarField(name="content_type", field_type="VARCHAR", max_length=50, 
                    description="Type of content (text, table, image, etc.)"),
            ScalarField(name="chunk_type", field_type="VARCHAR", max_length=50, 
                    description="Type of chunk (original, summary, etc.)"),
            ScalarField(name="page_number", field_type="INT64", 
                    description="Page number in the original document"),
            ScalarField(name="tree_level", field_type="INT64", 
                    description="RAPTOR tree level (0 for original chunks)"),
            ScalarField(name="metadata", field_type="JSON", 
                    description="Additional metadata about the chunk"),
        ]
        
        vector_field = VectorField(
            name="embedding",
            dimension=dimension
        )
        
        sparse_field = SparseField(
            name="sparse"
        )
        
        bm25_function = BM25Function(
            input_field="content",
            output_field="sparse"
        )
        
        partition_config = PartitionConfig(
            strategy="case_id",
            field="case_id"
        )
        
        return cls(
            name=collection_name,
            description="Document store with enhanced schema for vector and BM25 search",
            scalar_fields=scalar_fields,
            vector_field=vector_field,
            sparse_field=sparse_field,
            bm25_function=bm25_function,
            partition_config=partition_config
        )



class VectorEntity(BaseModel):
    """Entity to be stored in the vector database"""
    id: str
    document_id: str
    case_id: Optional[str] = "default"
    chunk_id: str
    content: str
    embedding: List[float]
    sparse_vector: Optional[Any] = None  # Add field for sparse vector (BM25)
    content_type: str = "text"  # "text", "table", "image", etc.
    chunk_type: str = "original"  # "original", "summary", etc.
    page_number: Optional[int] = None
    tree_level: Optional[int] = 0  # 0 for original chunks, > 0 for summaries
    metadata: Dict[str, Any] = {}
    
    def to_milvus_entity(self) -> Dict[str, Any]:
        """Convert to Milvus entity format"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "case_id": self.case_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "content_type": self.content_type,
            "chunk_type": self.chunk_type,
            "page_number": self.page_number,
            "tree_level": self.tree_level,
            "metadata": self.metadata,
            "embedding": self.embedding,
            # We don't need to include sparse_vector as it will be generated automatically by Milvus
        }
    
    @classmethod
    def from_milvus_entity(cls, entity: Dict[str, Any]) -> "VectorEntity":
        """Create VectorEntity from Milvus entity"""
        return cls(
            id=entity.get("id", ""),
            document_id=entity.get("document_id", ""),
            case_id=entity.get("case_id", "default"),
            chunk_id=entity.get("chunk_id", ""),
            content=entity.get("content", ""),
            embedding=entity.get("embedding", []),
            content_type=entity.get("content_type", "text"),
            chunk_type=entity.get("chunk_type", "original"),
            page_number=entity.get("page_number"),
            tree_level=entity.get("tree_level", 0),
            metadata=entity.get("metadata", {})
        )
        
class VectorSearchParams(BaseModel):
    """Parameters for vector search"""
    query_embedding: List[float]
    document_ids: Optional[List[str]] = None
    case_ids: Optional[List[str]] = None
    content_types: Optional[List[str]] = None
    chunk_types: Optional[List[str]] = None
    tree_levels: Optional[List[int]] = None
    page_numbers: Optional[List[int]] = None
    filter_expr: Optional[str] = None
    top_k: int = 5
    metric_type: str = "COSINE"
    params: Dict[str, Any] = Field(
        default_factory=lambda: {"ef": 64}
    )
    
    def get_filter_expr(self) -> Optional[str]:
        """Generate Milvus filter expression based on parameters"""
        expressions = []
        
        if self.document_ids:
            if len(self.document_ids) == 1:
                expressions.append(f'document_id == "{self.document_ids[0]}"')
            else:
                doc_list = '", "'.join(self.document_ids)
                expressions.append(f'document_id in ["{doc_list}"]')
        
        if self.case_ids:
            if len(self.case_ids) == 1:
                expressions.append(f'case_id == "{self.case_ids[0]}"')
            else:
                case_list = '", "'.join(self.case_ids)
                expressions.append(f'case_id in ["{case_list}"]')
                
        if self.content_types:
            if len(self.content_types) == 1:
                expressions.append(f'content_type == "{self.content_types[0]}"')
            else:
                type_list = '", "'.join(self.content_types)
                expressions.append(f'content_type in ["{type_list}"]')
        
        if self.chunk_types:
            if len(self.chunk_types) == 1:
                expressions.append(f'chunk_type == "{self.chunk_types[0]}"')
            else:
                type_list = '", "'.join(self.chunk_types)
                expressions.append(f'chunk_type in ["{type_list}"]')
        
        if self.tree_levels is not None:
            if len(self.tree_levels) == 1:
                expressions.append(f'tree_level == {self.tree_levels[0]}')
            else:
                level_list = ', '.join(map(str, self.tree_levels))
                expressions.append(f'tree_level in [{level_list}]')
        
        if self.page_numbers:
            if len(self.page_numbers) == 1:
                expressions.append(f'page_number == {self.page_numbers[0]}')
            else:
                page_list = ', '.join(map(str, self.page_numbers))
                expressions.append(f'page_number in [{page_list}]')
        
        # Add custom filter expression if provided
        if self.filter_expr:
            expressions.append(f'({self.filter_expr})')
        
        # Combine all expressions with AND
        if expressions:
            return " && ".join(expressions)
        
        return None


class VectorSearchResult(BaseModel):
    """Result from vector search"""
    entity: VectorEntity
    score: float
    distance: float