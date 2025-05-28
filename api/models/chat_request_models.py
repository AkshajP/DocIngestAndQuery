from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Optional, Any
from datetime import datetime
from db.chat_store.models import ChatSettings

class QueryContextRequest(BaseModel):
    """Enhanced query request that uses chat settings as primary configuration"""
    question: str = Field(..., description="User question")
    
    # Optional overrides for chat settings
    use_tree_search: Optional[bool] = Field(None, description="Override chat setting for tree search")
    use_hybrid_search: Optional[bool] = Field(None, description="Override chat setting for hybrid search")
    vector_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override chat setting for vector weight")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Override chat setting for top_k")
    tree_level_filter: Optional[List[int]] = Field(None, description="Override chat setting for tree levels")
    llm_model: Optional[str] = Field(None, description="Override chat setting for LLM model")
    
    # Stream preference
    stream: Optional[bool] = Field(True, description="Whether to stream the response")
    
    def merge_with_chat_settings(self, chat_settings: ChatSettings) -> ChatSettings:
        """Merge request overrides with chat settings, returning effective settings"""
        # Create a new ChatSettings object from the existing one
        if chat_settings:
            effective_settings = ChatSettings(**chat_settings.dict())
        else:
            effective_settings = ChatSettings()
        
        # Apply overrides where provided
        if self.use_tree_search is not None:
            effective_settings.use_tree_search = self.use_tree_search
        if self.use_hybrid_search is not None:
            effective_settings.use_hybrid_search = self.use_hybrid_search
        if self.vector_weight is not None:
            effective_settings.vector_weight = self.vector_weight
        if self.top_k is not None:
            effective_settings.top_k = self.top_k
        if self.tree_level_filter is not None:
            effective_settings.tree_level_filter = self.tree_level_filter
        if self.llm_model is not None:
            effective_settings.llm_model = self.llm_model
            
        return effective_settings

class RegenerateContextRequest(BaseModel):
    """Request to regenerate response with optional setting overrides"""
    # Same override pattern as QueryContextRequest
    use_tree_search: Optional[bool] = None
    use_hybrid_search: Optional[bool] = None
    vector_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=50)
    tree_level_filter: Optional[List[int]] = None
    llm_model: Optional[str] = None
    
    def merge_with_chat_settings(self, chat_settings: ChatSettings) -> ChatSettings:
        """Merge request overrides with chat settings"""
        # Create a new ChatSettings object from the existing one
        if chat_settings:
            effective_settings = ChatSettings(**chat_settings.dict())
        else:
            effective_settings = ChatSettings()
        
        if self.use_tree_search is not None:
            effective_settings.use_tree_search = self.use_tree_search
        if self.use_hybrid_search is not None:
            effective_settings.use_hybrid_search = self.use_hybrid_search
        if self.vector_weight is not None:
            effective_settings.vector_weight = self.vector_weight
        if self.top_k is not None:
            effective_settings.top_k = self.top_k
        if self.tree_level_filter is not None:
            effective_settings.tree_level_filter = self.tree_level_filter
        if self.llm_model is not None:
            effective_settings.llm_model = self.llm_model
            
        return effective_settings

class ChatContextRequest(BaseModel):
    """Base request for chat operations with access control context"""
    # This will be populated by dependency injection
    user_id: Optional[str] = None
    case_id: Optional[str] = None
    
    class Config:
        # Allow population by field name for dependency injection
        populate_by_name = True