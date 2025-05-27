import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from core.config import get_config
from db.document_store.repository import DocumentMetadataRepository
from services.document.storage import LocalStorageAdapter, S3StorageAdapter
from .processing_state_manager import ProcessingStateManager, ProcessingStage
from .stage_processors import (
    ExtractionProcessor, ChunkingProcessor, EmbeddingProcessor,
    TreeBuildingProcessor, VectorStorageProcessor
)

logger = logging.getLogger(__name__)

class PersistentUploadService:
    """
    Service for persistent document upload and processing.
    Manages document processing through stages with retry capabilities.
    """
    
    def __init__(self, config=None):
        """Initialize the persistent upload service"""
        self.config = config or get_config()
        
        # Initialize storage adapter
        self.storage_adapter = self._initialize_storage_adapter()
        
        # Initialize document metadata repository
        self.doc_repository = DocumentMetadataRepository()
        
        # Initialize stage processors
        self.processors = {
            ProcessingStage.EXTRACTION.value: ExtractionProcessor(self.config),
            ProcessingStage.CHUNKING.value: ChunkingProcessor(self.config),
            ProcessingStage.EMBEDDING.value: EmbeddingProcessor(self.config),
            ProcessingStage.TREE_BUILDING.value: TreeBuildingProcessor(self.config),
            ProcessingStage.VECTOR_STORAGE.value: VectorStorageProcessor(self.config)
        }
        
        logger.info("Initialized persistent upload service")
    
    def _initialize_storage_adapter(self):
        """Initialize the appropriate storage adapter based on configuration"""
        storage_type = self.config.storage.storage_type.lower()
        
        if storage_type == "s3":
            logger.info(f"Initializing S3 storage adapter with bucket {self.config.storage.s3_bucket}")
            return S3StorageAdapter(
                bucket_name=self.config.storage.s3_bucket,
                prefix=self.config.storage.s3_prefix,
                region=self.config.storage.aws_region
            )
        else:
            logger.info(f"Initializing local storage adapter with directory {self.config.storage.storage_dir}")
            return LocalStorageAdapter()
    
    def upload_document(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        case_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        force_restart: bool = False
    ) -> Dict[str, Any]:
        """
        Upload and process a document through the complete pipeline with persistence.
        
        Args:
            file_path: Path to the document file
            document_id: Optional custom ID (generated if not provided)
            case_id: Case ID for document grouping (required)
            metadata: Optional metadata about the document
            force_restart: If True, restart processing from the beginning
            
        Returns:
            Dictionary with processing status and information
        """
        process_start_time = time.time()
        
        # Generate document ID if not provided
        if not document_id:
            timestamp = int(time.time())
            base_name = os.path.basename(file_path)
            safe_name = ''.join(c for c in base_name.split('.')[0].replace(' ', '_') 
                              if c.isalnum() or c == '_')
            document_id = f"doc_{timestamp}_{safe_name}"
        
        try:
            # Initialize or load existing processing state
            result = self._initialize_document_processing(
                document_id, case_id, file_path, metadata, force_restart
            )
            
            if result["status"] == "error":
                return result
            
            doc_dir = result["doc_dir"]
            stored_file_path = result["stored_file_path"]
            
            # Initialize processing state manager
            state_manager = ProcessingStateManager(
                document_id=document_id,
                storage_adapter=self.storage_adapter,
                doc_dir=doc_dir
            )
            
            # Reset processing if force_restart is True
            if force_restart:
                state_manager.reset_to_stage(ProcessingStage.UPLOAD.value)
                logger.info(f"Force restarting processing for document {document_id}")
            
            # Create processing context
            context = {
                "document_id": document_id,
                "case_id": case_id,
                "file_path": file_path,
                "stored_file_path": stored_file_path,
                "doc_dir": doc_dir,
                "storage_adapter": self.storage_adapter,
                "metadata": metadata or {}
            }
            
            # Execute processing pipeline
            pipeline_result = self._execute_processing_pipeline(state_manager, context)
            
            # Calculate total processing time
            total_processing_time = time.time() - process_start_time
            
            # Update final document metadata
            final_metadata = {
                "total_processing_time": total_processing_time,
                "processing_state": state_manager.get_stage_status()
            }
            
            if pipeline_result["status"] == "success":
                final_metadata.update({
                    "status": "processed",
                    "processing_date": datetime.now().isoformat()
                })
                
                # Mark processing as completed
                state_manager.mark_stage_complete(ProcessingStage.COMPLETED.value)
            else:
                final_metadata.update({
                    "status": "failed",
                    "failure_stage": pipeline_result.get("failed_stage"),
                    "error_message": pipeline_result.get("error"),
                    "failure_time": datetime.now().isoformat()
                })
            
            self.doc_repository.update_document(document_id, final_metadata)
            
            # Prepare response
            response = {
                "status": pipeline_result["status"],
                "document_id": document_id,
                "case_id": case_id,
                "processing_time": total_processing_time,
                "stored_file_path": stored_file_path,
                "doc_dir": doc_dir,
                "processing_state": state_manager.get_stage_status()
            }
            
            if pipeline_result["status"] == "success":
                response.update({
                    "chunks_count": pipeline_result.get("chunks_count", 0),
                    "tree_nodes_count": pipeline_result.get("tree_nodes_count", 0),
                    "raptor_levels": pipeline_result.get("raptor_levels", [])
                })
            else:
                response.update({
                    "error": pipeline_result.get("error"),
                    "failed_stage": pipeline_result.get("failed_stage")
                })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            
            # Update document with error status
            self.doc_repository.update_document(document_id, {
                "status": "failed",
                "failure_stage": "initialization",
                "error_message": str(e),
                "failure_time": datetime.now().isoformat()
            })
            
            return {
                "status": "error",
                "document_id": document_id,
                "error": str(e),
                "processing_time": time.time() - process_start_time
            }
    
    def retry_document_processing(
        self,
        document_id: str,
        from_stage: Optional[str] = None,
        case_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retry document processing from a specific stage or current failed stage.
        
        Args:
            document_id: Document ID to retry
            from_stage: Optional stage to retry from (if None, uses current stage)
            case_id: Optional case ID for verification
            
        Returns:
            Dictionary with retry status and information
        """
        try:
            # Get document metadata
            document = self.doc_repository.get_document(document_id)
            if not document:
                return {
                    "status": "error",
                    "error": "Document not found"
                }
            
            # Verify case_id if provided
            if case_id and document.get("case_id") != case_id:
                return {
                    "status": "error",
                    "error": "Document does not belong to specified case"
                }
            
            # Get document paths
            file_path = document.get("stored_file_path") or document.get("original_file_path")
            if not file_path or not os.path.exists(file_path):
                return {
                    "status": "error",
                    "error": "Original file not found"
                }
            
            doc_dir = os.path.join(self.config.storage.storage_dir, document_id)
            
            # Initialize processing state manager
            state_manager = ProcessingStateManager(
                document_id=document_id,
                storage_adapter=self.storage_adapter,
                doc_dir=doc_dir
            )
            
            # Reset to specified stage if provided
            if from_stage:
                if not self._is_valid_stage(from_stage):
                    return {
                        "status": "error",
                        "error": f"Invalid stage: {from_stage}"
                    }
                state_manager.reset_to_stage(from_stage)
                logger.info(f"Reset document {document_id} to stage {from_stage}")
            
            # Update document status to processing
            self.doc_repository.update_document(document_id, {
                "status": "processing",
                "retry_time": datetime.now().isoformat(),
                "retry_count": document.get("retry_count", 0) + 1
            })
            
            # Create processing context
            context = {
                "document_id": document_id,
                "case_id": document.get("case_id", "default"),
                "file_path": file_path,
                "stored_file_path": file_path,
                "doc_dir": doc_dir,
                "storage_adapter": self.storage_adapter,
                "metadata": document.get("user_metadata", {})
            }
            
            # Execute processing pipeline
            pipeline_result = self._execute_processing_pipeline(state_manager, context)
            
            # Update document metadata based on result
            if pipeline_result["status"] == "success":
                self.doc_repository.update_document(document_id, {
                    "status": "processed",
                    "processing_date": datetime.now().isoformat(),
                    "processing_state": state_manager.get_stage_status()
                })
                state_manager.mark_stage_complete(ProcessingStage.COMPLETED.value)
            else:
                self.doc_repository.update_document(document_id, {
                    "status": "failed",
                    "failure_stage": pipeline_result.get("failed_stage"),
                    "error_message": pipeline_result.get("error"),
                    "failure_time": datetime.now().isoformat(),
                    "processing_state": state_manager.get_stage_status()
                })
            
            return {
                "status": pipeline_result["status"],
                "document_id": document_id,
                "processing_state": state_manager.get_stage_status(),
                **pipeline_result
            }
            
        except Exception as e:
            logger.error(f"Error retrying document {document_id}: {str(e)}")
            return {
                "status": "error",
                "document_id": document_id,
                "error": str(e)
            }
    
    def get_document_processing_status(self, document_id: str) -> Dict[str, Any]:
        """Get detailed processing status for a document"""
        try:
            # Get document metadata
            document = self.doc_repository.get_document(document_id)
            if not document:
                return {
                    "status": "error",
                    "error": "Document not found"
                }
            
            doc_dir = os.path.join(self.config.storage.storage_dir, document_id)
            
            # Initialize processing state manager
            state_manager = ProcessingStateManager(
                document_id=document_id,
                storage_adapter=self.storage_adapter,
                doc_dir=doc_dir
            )
            
            # Get comprehensive status
            processing_status = state_manager.get_stage_status()
            
            return {
                "status": "success",
                "document_metadata": document,
                "processing_status": processing_status
            }
            
        except Exception as e:
            logger.error(f"Error getting processing status for {document_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _initialize_document_processing(
        self,
        document_id: str,
        case_id: str,
        file_path: str,
        metadata: Optional[Dict[str, Any]],
        force_restart: bool
    ) -> Dict[str, Any]:
        """Initialize document processing setup"""
        try:
            # Create document directory path
            doc_dir = os.path.join(self.config.storage.storage_dir, document_id)
            target_filename = "original.pdf"
            target_path = os.path.join(doc_dir, target_filename)
            
            # Create the directory if it doesn't exist
            self.storage_adapter.create_directory(doc_dir)
            
            # Check if document already exists in registry
            existing_doc = self.doc_repository.get_document(document_id)
            
            if existing_doc and not force_restart:
                # Document exists, return existing paths
                stored_file_path = existing_doc.get("stored_file_path", target_path)
                logger.info(f"Document {document_id} already exists, using existing setup")
                return {
                    "status": "success",
                    "doc_dir": doc_dir,
                    "stored_file_path": stored_file_path
                }
            
            # Store the original file
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            storage_success = self.storage_adapter.write_file(file_content, target_path)
            
            if not storage_success:
                logger.warning(f"Failed to save original PDF to {target_path}")
                stored_file_path = file_path  # Fall back to original path
            else:
                logger.info(f"Successfully saved original PDF to {target_path}")
                stored_file_path = target_path
            
            # Initialize or update document metadata
            doc_metadata = {
                "document_id": document_id,
                "case_id": case_id,
                "original_filename": os.path.basename(file_path),
                "original_file_path": file_path,
                "stored_file_path": stored_file_path,
                "file_type": os.path.splitext(file_path)[1].lower()[1:],
                "processing_start_time": datetime.now().isoformat(),
                "status": "processing",
                "user_metadata": metadata or {}
            }
            
            if existing_doc:
                self.doc_repository.update_document(document_id, doc_metadata)
            else:
                self.doc_repository.add_document(doc_metadata)
            
            return {
                "status": "success",
                "doc_dir": doc_dir,
                "stored_file_path": stored_file_path
            }
            
        except Exception as e:
            logger.error(f"Error initializing document processing: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _execute_processing_pipeline(
        self,
        state_manager: ProcessingStateManager,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the processing pipeline from current stage"""
        
        # Define stage execution order
        stage_order = [
            ProcessingStage.EXTRACTION.value,
            ProcessingStage.CHUNKING.value,
            ProcessingStage.EMBEDDING.value,
            ProcessingStage.TREE_BUILDING.value,
            ProcessingStage.VECTOR_STORAGE.value
        ]
        
        current_stage = state_manager.get_current_stage()
        completed_stages = state_manager.get_completed_stages()
        
        # Find starting point in pipeline
        start_index = 0
        if current_stage in stage_order:
            start_index = stage_order.index(current_stage)
        
        logger.info(f"Starting processing pipeline for document {context['document_id']} from stage {current_stage}")
        
        # Execute stages in order
        last_result = {}
        for i in range(start_index, len(stage_order)):
            stage = stage_order[i]
            
            # Skip completed stages
            if stage in completed_stages:
                logger.info(f"Skipping completed stage: {stage}")
                continue
            
            processor = self.processors.get(stage)
            if not processor:
                error_msg = f"No processor found for stage: {stage}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "failed_stage": stage
                }
            
            # Check if stage can be executed
            if not processor.can_execute(state_manager, context):
                error_msg = f"Stage {stage} cannot be executed"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "failed_stage": stage
                }
            
            # Validate dependencies
            if not processor.validate_dependencies(state_manager, context):
                error_msg = f"Dependencies not met for stage {stage}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "failed_stage": stage
                }
            
            # Execute stage
            logger.info(f"Executing stage: {stage}")
            result = processor.execute(state_manager, context)
            
            if result["status"] != "success":
                logger.error(f"Stage {stage} failed: {result.get('error')}")
                return {
                    "status": "error",
                    "error": result.get("error"),
                    "failed_stage": stage,
                    "processing_time": result.get("processing_time", 0)
                }
            
            last_result = result
            logger.info(f"Stage {stage} completed successfully")
        
        logger.info(f"Processing pipeline completed successfully for document {context['document_id']}")
        
        return {
            "status": "success",
            **last_result
        }
    
    def _is_valid_stage(self, stage: str) -> bool:
        """Check if a stage name is valid"""
        valid_stages = [s.value for s in ProcessingStage]
        return stage in valid_stages