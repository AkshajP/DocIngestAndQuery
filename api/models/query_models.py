# api/models/query_models.py

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

class QueryContext(BaseModel):
    """Context for a query, including settings and state"""
    chat_id: Optional[str] = None
    use_tree: bool = False
    use_history: bool = True
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    layer: Optional[int] = None  # For tree-based retrieval
    tree_level_filter: Optional[List[int]] = None  # Filter by tree level
    model_override: Optional[str] = None  # Override default model
    document_ids: List[str] = []


class QueryProcessingStats(BaseModel):
    """Processing statistics for a query"""
    time_taken: float
    input_tokens: int
    output_tokens: int
    total_tokens: int  # Combined token count
    retrieval_time: Optional[float] = None
    llm_time: Optional[float] = None
    method: str = "flat"  # "flat" or "tree"
    model_used: Optional[str] = None  # Model used for this query


class QueryRequest(BaseModel):
    """Request to submit a query"""
    question: str
    use_tree: Optional[bool] = False
    top_k: Optional[int] = Field(5, ge=1, le=20)
    tree_level_filter: Optional[List[int]] = None  # Filter by tree level
    model_override: Optional[str] = None  # Override default model
    
    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v):
        if v is not None and (v < 1 or v > 20):
            raise ValueError('top_k must be between 1 and 20')
        return v
    
    @field_validator('tree_level_filter')
    @classmethod
    def validate_tree_level_filter(cls, v):
        if v is not None:
            # Ensure all values are non-negative integers
            if not all(isinstance(level, int) and level >= 0 for level in v):
                raise ValueError('tree_level_filter must contain non-negative integers')
        return v

class RegenerateResponseRequest(BaseModel):
    """Request to regenerate a response"""
    use_tree: Optional[bool] = False
    top_k: Optional[int] = Field(5, ge=1, le=20)
    tree_level_filter: Optional[List[int]] = None  # Filter by tree level
    model_override: Optional[str] = None  # Override default model
    
    # Add the same validators here as well
    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v):
        if v is not None and (v < 1 or v > 20):
            raise ValueError('top_k must be between 1 and 20')
        return v
    
    @field_validator('tree_level_filter')
    @classmethod
    def validate_tree_level_filter(cls, v):
        if v is not None:
            # Ensure all values are non-negative integers
            if not all(isinstance(level, int) and level >= 0 for level in v):
                raise ValueError('tree_level_filter must contain non-negative integers')
        return v

class QuerySource(BaseModel):
    """Source information for a query response"""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = {}
    original_boxes: Optional[List[Dict[str, Any]]] = None
    tree_level: Optional[int] = 0  # Tree level (0 = original, >0 = summary)

class QueryResponse(BaseModel):
    """Response to a query"""
    id: str
    answer: str
    sources: List[QuerySource]
    processing_stats: QueryProcessingStats

    
