"""
Main chat routes module that combines query and management endpoints.

This module provides a unified interface for all chat-related operations,
split into two logical groups:
1. Query operations (submit_query, regenerate, highlights)
2. Management operations (CRUD, settings, documents)
"""

from fastapi import APIRouter

# Import the split route modules
from api.routes import chat_query_routes, chat_management_routes

# Create the main router
router = APIRouter()

# Include both sub-routers
router.include_router(
    chat_query_routes.router,
    tags=["chat-queries"]
)

router.include_router(
    chat_management_routes.router, 
    tags=["chat-management"]
)

# Export for use in main app
__all__ = ["router"]