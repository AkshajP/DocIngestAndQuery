import sys
sys.dont_write_bytecode = True
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from core.service_manager import get_service_manager
from api.routes import document_routes, admin_routes
import logging 
from api.routes import chat_query_routes, chat_management_routes, test_celery_routes
logging.getLogger().setLevel(logging.DEBUG)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown tasks"""
    # Startup
    # logger.info("Starting application...")
    
    try:
        # Initialize all services
        service_manager = get_service_manager()
        service_manager.initialize()
        # logger.info("Application services initialized successfully")
        
        yield
        
    finally:
        # Shutdown
        # logger.info("Shutting down application...")
        service_manager = get_service_manager()
        service_manager.shutdown()
        # logger.info("Application shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Document LLM API",
    description="API for document-based LLM interactions",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your Next.js app
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include your chat router
app.include_router(
    chat_query_routes.router,
    tags=["chat-queries"]
)

app.include_router(
    chat_management_routes.router, 
    tags=["chat-management"]
)
# app.include_router(chat_routes.router)
app.include_router(document_routes.router)
app.include_router(admin_routes.router) 
app.include_router(test_celery_routes.router)


@app.get("/")
async def read_root():
    return {"message": "FastAPI server is running!"}