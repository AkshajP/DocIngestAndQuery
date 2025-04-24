# api/models/query_models.py

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime


class QueryContext(BaseModel):
    """Context for a query, including settings and state"""
    chat_id: Optional[str] = None
    use_tree: bool = False
    use_history: bool = True
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    layer: Optional[int] = None  # For tree-based retrieval
    document_ids: List[str] = []


class QuerySource(BaseModel):
    """Source information for a query response"""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = {}
    original_boxes: Optional[List[Dict[str, Any]]] = None


class QueryProcessingStats(BaseModel):
    """Processing statistics for a query"""
    time_taken: float
    tokens_used: int
    retrieval_time: Optional[float] = None
    llm_time: Optional[float] = None
    method: str = "flat"  # "flat" or "tree"


class QueryRequest(BaseModel):
    """Request to submit a query"""
    question: str
    use_tree: Optional[bool] = False
    top_k: Optional[int] = Field(5, ge=1, le=20)
    layer: Optional[int] = None
    
    @validator('top_k')
    def validate_top_k(cls, v):
        if v is not None and (v < 1 or v > 20):
            raise ValueError('top_k must be between 1 and 20')
        return v


class QueryResponse(BaseModel):
    """Response to a query"""
    id: str
    answer: str
    sources: List[QuerySource]
    processing_stats: QueryProcessingStats


class RegenerateResponseRequest(BaseModel):
    """Request to regenerate a response"""
    use_tree: Optional[bool] = False
    top_k: Optional[int] = Field(5, ge=1, le=20)
    layer: Optional[int] = None