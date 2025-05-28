import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from services.document.storage import StorageAdapter
from services.pdf.extractor import PDFExtractor
from services.document.chunker import Chunker
from services.ml.embeddings import EmbeddingService
from services.retrieval.raptor import Raptor
from db.vector_store.adapter import VectorStoreAdapter
from .processing_state_manager import ProcessingStateManager, ProcessingStage

logger = logging.getLogger(__name__)

class StageProcessor(ABC):
    """Base class for stage processors"""
    
    def __init__(self, config=None):
        self.config = config
    
    @abstractmethod
    def can_execute(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> bool:
        """Check if this stage can be executed given current state"""
        pass
    
    @abstractmethod
    def execute(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the stage processing"""
        pass
    
    @abstractmethod
    def validate_dependencies(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> bool:
        """Validate that required previous stages are completed"""
        pass
    
    def get_stage_name(self) -> str:
        """Get the stage name this processor handles"""
        return self.__class__.__name__.replace('Processor', '').lower()

class ExtractionProcessor(StageProcessor):
    """Processor for document content extraction stage"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.pdf_extractor = PDFExtractor(language=config.processing.language if config else 'en')
    
    def can_execute(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> bool:
        """Check if extraction can be executed"""
        current_stage = state_manager.get_current_stage()
        return (current_stage == ProcessingStage.UPLOAD.value or 
                current_stage == ProcessingStage.EXTRACTION.value)
    
    def validate_dependencies(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> bool:
        """Validate that file upload is completed"""
        # Check if original file exists
        file_path = context.get("file_path") or context.get("stored_file_path")
        if not file_path or not os.path.exists(file_path):
            logger.error(f"Original file not found for document {state_manager.document_id}: {file_path}")
            return False
        return True
    
    def execute(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content extraction"""
        start_time = time.time()
        
        try:
            # Get file path from context
            file_path = context.get("file_path") or context.get("stored_file_path")
            if not file_path:
                raise ValueError("No file path provided in context")
            
            # Validate file exists
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            
            # Create document-specific images directory
            images_dir = os.path.join(state_manager.doc_dir, "images")
            storage_adapter = context.get("storage_adapter")
            if storage_adapter:
                storage_adapter.create_directory(images_dir)
            
            logger.info(f"Starting content extraction for document {state_manager.document_id}")
            
            # Extract content and save images to document-specific directory
            extraction_result = self.pdf_extractor.extract_content(
                file_path, 
                save_images=True,
                output_dir=images_dir,
                storage_adapter=storage_adapter
            )
            
            if extraction_result["status"] != "success":
                raise Exception(extraction_result.get("message", "Extraction failed"))
            
            # Save extraction results
            content_list = extraction_result["content_list"]
            success = state_manager.save_stage_data(
                ProcessingStage.EXTRACTION.value, 
                {
                    "content_list": content_list,
                    "page_count": extraction_result.get("page_count", 0),
                    "images": extraction_result.get("images", []),
                    "extraction_metadata": {
                        "method": "pdf_extractor",
                        "images_directory": images_dir,
                        "images_count": len(extraction_result.get("images", []))
                    }
                },
                "extraction_result"
            )
            
            if not success:
                raise Exception("Failed to save extraction results")
            
            # Mark stage as complete
            state_manager.mark_stage_complete(ProcessingStage.EXTRACTION.value)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Content extraction completed for document {state_manager.document_id} in {processing_time:.2f}s")
            
            return {
                "status": "success",
                "content_list": content_list,
                "page_count": extraction_result.get("page_count", 0),
                "processing_time": processing_time,
                "images_directory": images_dir,
                "images_count": len(extraction_result.get("images", []))
            }
            
        except Exception as e:
            error_msg = f"Content extraction failed: {str(e)}"
            state_manager.mark_stage_failed(ProcessingStage.EXTRACTION.value, error_msg)
            logger.error(f"Extraction failed for document {state_manager.document_id}: {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "processing_time": time.time() - start_time
            }

class ChunkingProcessor(StageProcessor):
    """Processor for document chunking stage"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.chunker = Chunker(
            max_chunk_size=config.processing.chunk_size if config else 5000,
            min_chunk_size=200,
            overlap_size=config.processing.chunk_overlap if config else 200
        )
    
    def can_execute(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> bool:
        """Check if chunking can be executed"""
        current_stage = state_manager.get_current_stage()
        
        # If we're still in extraction stage but it's complete, advance to chunking
        if (current_stage == ProcessingStage.EXTRACTION.value and 
            state_manager.is_stage_complete(ProcessingStage.EXTRACTION.value)):
            # Advance to chunking stage
            state_manager.processing_state["current_stage"] = ProcessingStage.CHUNKING.value
            state_manager._save_processing_state()
            logger.info(f"Advanced document {state_manager.document_id} from extraction to chunking stage")
            return True
        
        # If already in chunking stage, we can execute
        return current_stage == ProcessingStage.CHUNKING.value

    def validate_dependencies(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> bool:
        """Validate that extraction is completed"""
        if not state_manager.is_stage_complete(ProcessingStage.EXTRACTION.value):
            logger.error(f"Extraction stage not completed for document {state_manager.document_id}")
            return False
        
        # Check if extraction data exists and is valid
        extraction_data = state_manager.load_stage_data(ProcessingStage.EXTRACTION.value, "extraction_result")
        if not extraction_data:
            logger.error(f"No extraction data found for document {state_manager.document_id}")
            return False
        
        # Validate content_list structure
        content_list = extraction_data.get("content_list")
        if not content_list or not isinstance(content_list, list):
            logger.error(f"Invalid or empty content_list for document {state_manager.document_id}")
            return False
        
        # Check if content_list has valid items
        valid_items = [item for item in content_list if item.get("content") or item.get("text")]
        if not valid_items:
            logger.error(f"No valid content items found for document {state_manager.document_id}")
            return False
        
        logger.info(f"Validation passed: found {len(valid_items)} valid content items")
        return True

    def execute(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute chunking with improved error handling"""
        start_time = time.time()
        
        try:
            # Load extraction results with validation
            extraction_data = state_manager.load_stage_data(ProcessingStage.EXTRACTION.value, "extraction_result")
            if not extraction_data:
                raise ValueError("No extraction data found")
            
            content_list = extraction_data.get("content_list", [])
            if not content_list:
                raise ValueError("Content list is empty")
            
            logger.info(f"Starting chunking for document {state_manager.document_id} with {len(content_list)} content items")
            
            # Process content using the chunker
            chunks = self.chunker.chunk_content(content_list)
            
            if not chunks:
                raise ValueError("No chunks created from document content")
            
            # Validate chunk structure
            valid_chunks = []
            for i, chunk in enumerate(chunks):
                if not chunk.get("content"):
                    logger.warning(f"Chunk {i} has no content, skipping")
                    continue
                if not chunk.get("id"):
                    chunk["id"] = f"chunk_{state_manager.document_id}_{i}"
                valid_chunks.append(chunk)
            
            if not valid_chunks:
                raise ValueError("No valid chunks created")
            
            # Count content types
            content_types = {}
            for chunk in valid_chunks:
                chunk_type = chunk.get("metadata", {}).get("type", "unknown")
                content_types[chunk_type] = content_types.get(chunk_type, 0) + 1
            
            # Save chunking results
            chunking_data = {
                "chunks": valid_chunks,
                "chunks_count": len(valid_chunks),
                "content_types": content_types,
                "chunking_metadata": {
                    "max_chunk_size": self.chunker.max_chunk_size,
                    "overlap_size": self.chunker.overlap_size,
                    "original_content_items": len(content_list),
                    "processing_timestamp": datetime.now().isoformat()
                }
            }
            
            success = state_manager.save_stage_data(
                ProcessingStage.CHUNKING.value,
                chunking_data,
                "chunks"
            )
            
            if not success:
                raise RuntimeError("Failed to save chunking results to storage")
            
            # Mark stage as complete
            state_manager.mark_stage_complete(ProcessingStage.CHUNKING.value)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Chunking completed for document {state_manager.document_id} in {processing_time:.2f}s: {len(valid_chunks)} chunks created")
            
            return {
                "status": "success",
                "chunks": valid_chunks,
                "chunks_count": len(valid_chunks),
                "content_types": content_types,
                "processing_time": processing_time
            }
            
        except Exception as e:
            error_msg = f"Chunking failed: {str(e)}"
            state_manager.mark_stage_failed(ProcessingStage.CHUNKING.value, error_msg)
            logger.error(f"Chunking failed for document {state_manager.document_id}: {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "processing_time": time.time() - start_time
            }

class EmbeddingProcessor(StageProcessor):
    """Processor for embedding generation stage"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.embedding_service = EmbeddingService(
            model_name=config.ollama.embed_model if config else "llama3.2",
            base_url=config.ollama.base_url if config else "http://localhost:11434"
        )
    
    def can_execute(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> bool:
        """Check if embedding generation can be executed"""
        current_stage = state_manager.get_current_stage()
        return current_stage == ProcessingStage.EMBEDDING.value
    
    def validate_dependencies(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> bool:
        """Validate that chunking is completed"""
        if not state_manager.is_stage_complete(ProcessingStage.CHUNKING.value):
            logger.error(f"Chunking stage not completed for document {state_manager.document_id}")
            return False
        
        # Check if chunking data exists
        chunking_data = state_manager.load_stage_data(ProcessingStage.CHUNKING.value, "chunks")
        if not chunking_data or not chunking_data.get("chunks"):
            logger.error(f"No chunking data found for document {state_manager.document_id}")
            return False
        
        return True
    
    def execute(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute embedding generation"""
        start_time = time.time()
        
        try:
            # Load chunking results
            chunking_data = state_manager.load_stage_data(ProcessingStage.CHUNKING.value, "chunks")
            if not chunking_data:
                raise Exception("No chunking data found")
            
            chunks = chunking_data["chunks"]
            
            logger.info(f"Starting embedding generation for {len(chunks)} chunks in document {state_manager.document_id}")
            
            # Prepare texts dictionary with chunk IDs and contents
            texts_dict = {chunk["id"]: chunk["content"] for chunk in chunks}
            
            # Generate embeddings
            chunk_embeddings = self.embedding_service.generate_embeddings_dict(
                texts_dict, 
                show_progress=True
            )
            
            if not chunk_embeddings:
                raise Exception("Failed to generate embeddings")
            
            # Save embedding results
            embedding_data = {
                "embeddings": chunk_embeddings,
                "embedding_metadata": {
                    "model_name": self.embedding_service.model_name,
                    "embedding_dimension": self.embedding_service.embedding_dim,
                    "chunks_count": len(chunks),
                    "embeddings_count": len(chunk_embeddings)
                }
            }
            
            success = state_manager.save_stage_data(
                ProcessingStage.EMBEDDING.value,
                embedding_data,
                "embeddings"
            )
            
            if not success:
                raise Exception("Failed to save embedding results")
            
            # Mark stage as complete
            state_manager.mark_stage_complete(ProcessingStage.EMBEDDING.value)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Embedding generation completed for document {state_manager.document_id} in {processing_time:.2f}s")
            
            return {
                "status": "success",
                "embeddings": chunk_embeddings,
                "processing_time": processing_time
            }
            
        except Exception as e:
            error_msg = f"Embedding generation failed: {str(e)}"
            state_manager.mark_stage_failed(ProcessingStage.EMBEDDING.value, error_msg)
            logger.error(f"Embedding generation failed for document {state_manager.document_id}: {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "processing_time": time.time() - start_time
            }

class TreeBuildingProcessor(StageProcessor):
    """Processor for RAPTOR tree building stage"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.raptor = Raptor(
            ollama_base_url=config.ollama.base_url if config else "http://localhost:11434",
            ollama_model=config.ollama.model if config else "llama3.2",
            ollama_embed_model=config.ollama.embed_model if config else "llama3.2",
            max_tree_levels=config.processing.max_tree_levels if config else 3
        )
        self.embedding_service = EmbeddingService(
            model_name=config.ollama.embed_model if config else "llama3.2",
            base_url=config.ollama.base_url if config else "http://localhost:11434"
        )
    
    def can_execute(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> bool:
        """Check if tree building can be executed"""
        current_stage = state_manager.get_current_stage()
        return current_stage == ProcessingStage.TREE_BUILDING.value
    
    def validate_dependencies(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> bool:
        """Validate that embedding generation is completed"""
        if not state_manager.is_stage_complete(ProcessingStage.EMBEDDING.value):
            logger.error(f"Embedding stage not completed for document {state_manager.document_id}")
            return False
        
        # Check if required data exists
        chunking_data = state_manager.load_stage_data(ProcessingStage.CHUNKING.value, "chunks")
        embedding_data = state_manager.load_stage_data(ProcessingStage.EMBEDDING.value, "embeddings")
        
        if not chunking_data or not embedding_data:
            logger.error(f"Required data not found for tree building in document {state_manager.document_id}")
            return False
        
        return True
    
    def execute(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAPTOR tree building"""
        start_time = time.time()
        
        try:
            # Load required data
            chunking_data = state_manager.load_stage_data(ProcessingStage.CHUNKING.value, "chunks")
            embedding_data = state_manager.load_stage_data(ProcessingStage.EMBEDDING.value, "embeddings")
            
            if not chunking_data or not embedding_data:
                raise Exception("Required data not found")
            
            chunks = chunking_data["chunks"]
            chunk_embeddings = embedding_data["embeddings"]
            
            logger.info(f"Starting RAPTOR tree building for document {state_manager.document_id}")
            
            # Format chunks for RAPTOR
            chunk_texts = [chunk["content"] for chunk in chunks]
            chunk_ids = [chunk["id"] for chunk in chunks]
            content_types = [chunk.get("metadata", {}).get("type", "text") for chunk in chunks]
            
            # Build RAPTOR tree
            tree_data = self.raptor.build_tree(
                chunk_texts, 
                chunk_ids, 
                chunk_embeddings,
                content_types=content_types
            )
            
            if not tree_data:
                raise Exception("Failed to build RAPTOR tree")
            
            # Generate embeddings for tree nodes
            tree_nodes = {}
            for level, level_data in tree_data.items():
                if level <= 0:
                    continue
                    
                summaries_df = level_data.get("summaries_df")
                if summaries_df is None:
                    continue
                    
                for _, row in summaries_df.iterrows():
                    summary_text = row.get("summaries", "")
                    cluster_id = row.get("cluster")
                    
                    if not summary_text:
                        continue
                    
                    # Create node ID
                    node_id = f"summary_l{level}_c{cluster_id}"
                    tree_nodes[node_id] = summary_text
            
            # Generate embeddings for tree nodes
            tree_embeddings = self.embedding_service.generate_embeddings_dict(
                tree_nodes, 
                show_progress=True
            )
            
            # Convert tree_data DataFrames to JSON-serializable format
            json_serializable_tree_data = {}
            for level, level_data in tree_data.items():
                json_serializable_tree_data[level] = {}
                for key, value in level_data.items():
                    if hasattr(value, 'to_dict'):  # Check if it's a DataFrame
                        # Convert DataFrame to dictionary
                        json_serializable_tree_data[level][key] = value.to_dict('records')
                    else:
                        json_serializable_tree_data[level][key] = value
            
            # Save tree building results
            tree_building_data = {
                "tree_data": json_serializable_tree_data,
                "tree_metadata": {
                    "raptor_levels": sorted(list(tree_data.keys())),
                    "tree_nodes_count": len(tree_nodes),
                    "max_tree_levels": self.raptor.max_tree_levels
                }
            }
            
            # Save tree data and embeddings separately for better organization
            success1 = state_manager.save_stage_data(
                ProcessingStage.TREE_BUILDING.value,
                tree_building_data,
                "tree_data"
            )
            
            success2 = state_manager.save_stage_data(
                ProcessingStage.TREE_BUILDING.value,
                tree_embeddings,
                "tree_embeddings"
            )
            
            if not (success1 and success2):
                raise Exception("Failed to save tree building results")
            
            # Mark stage as complete
            state_manager.mark_stage_complete(ProcessingStage.TREE_BUILDING.value)
            
            processing_time = time.time() - start_time
            
            logger.info(f"RAPTOR tree building completed for document {state_manager.document_id} in {processing_time:.2f}s")
            
            return {
                "status": "success",
                "tree_data": tree_data,  # Return original format with DataFrames
                "tree_embeddings": tree_embeddings,
                "raptor_levels": sorted(list(tree_data.keys())),
                "processing_time": processing_time
            }
            
        except Exception as e:
            error_msg = f"Tree building failed: {str(e)}"
            state_manager.mark_stage_failed(ProcessingStage.TREE_BUILDING.value, error_msg)
            logger.error(f"Tree building failed for document {state_manager.document_id}: {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "processing_time": time.time() - start_time
            }

class VectorStorageProcessor(StageProcessor):
    """Processor for vector database storage stage"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.vector_store = VectorStoreAdapter(config=config)
    
    def can_execute(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> bool:
        """Check if vector storage can be executed"""
        current_stage = state_manager.get_current_stage()
        return current_stage == ProcessingStage.VECTOR_STORAGE.value
    
    def validate_dependencies(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> bool:
        """Validate that tree building is completed"""
        if not state_manager.is_stage_complete(ProcessingStage.TREE_BUILDING.value):
            logger.error(f"Tree building stage not completed for document {state_manager.document_id}")
            return False
        
        # Check if required data exists
        required_data = [
            (ProcessingStage.CHUNKING.value, "chunks"),
            (ProcessingStage.EMBEDDING.value, "embeddings"),
            (ProcessingStage.TREE_BUILDING.value, "tree_data"),
            (ProcessingStage.TREE_BUILDING.value, "tree_embeddings")
        ]
        
        for stage, filename in required_data:
            if not state_manager.load_stage_data(stage, filename):
                logger.error(f"Required data {stage}/{filename} not found for document {state_manager.document_id}")
                return False
        
        return True
    
    def execute(self, state_manager: ProcessingStateManager, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vector storage"""
        start_time = time.time()
        
        try:
            # Load all required data
            chunking_data = state_manager.load_stage_data(ProcessingStage.CHUNKING.value, "chunks")
            embedding_data = state_manager.load_stage_data(ProcessingStage.EMBEDDING.value, "embeddings")
            tree_data_result = state_manager.load_stage_data(ProcessingStage.TREE_BUILDING.value, "tree_data")
            tree_embeddings = state_manager.load_stage_data(ProcessingStage.TREE_BUILDING.value, "tree_embeddings")
            
            if not all([chunking_data, embedding_data, tree_data_result, tree_embeddings]):
                raise Exception("Required data not found")
            
            chunks = chunking_data["chunks"]
            chunk_embeddings = embedding_data["embeddings"]
            
            # Convert JSON-serializable tree data back to DataFrame format for vector storage
            json_tree_data = tree_data_result["tree_data"]
            tree_data = {}
            
            # Convert back to the expected format with DataFrames
            for level, level_data in json_tree_data.items():
                tree_data[int(level)] = {}  # Ensure level is integer
                for key, value in level_data.items():
                    if key == "summaries_df" and isinstance(value, list):
                        # Convert list of records back to DataFrame
                        import pandas as pd
                        tree_data[int(level)][key] = pd.DataFrame(value)
                    else:
                        tree_data[int(level)][key] = value
            
            document_id = state_manager.document_id
            case_id = context.get("case_id", "default")
            
            logger.info(f"Starting vector storage for document {document_id}")
            
            # Store original chunks
            chunks_count = self.vector_store.add_document_chunks(
                document_id, 
                case_id, 
                chunks, 
                chunk_embeddings
            )
            
            if chunks_count == 0:
                raise Exception("Failed to store document chunks in vector database")
            
            # Store tree nodes
            nodes_count = self.vector_store.add_tree_nodes(
                document_id, 
                case_id, 
                tree_data, 
                tree_embeddings
            )
            
            # Save storage results
            storage_data = {
                "chunks_stored": chunks_count,
                "tree_nodes_stored": nodes_count,
                "storage_metadata": {
                    "vector_store_type": "milvus",
                    "case_id": case_id,
                    "storage_timestamp": time.time()
                }
            }
            
            success = state_manager.save_stage_data(
                ProcessingStage.VECTOR_STORAGE.value,
                storage_data,
                "storage_result"
            )
            
            if not success:
                raise Exception("Failed to save storage results")
            
            # Mark stage as complete
            state_manager.mark_stage_complete(ProcessingStage.VECTOR_STORAGE.value)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Vector storage completed for document {document_id} in {processing_time:.2f}s")
            
            return {
                "status": "success",
                "chunks_count": chunks_count,
                "tree_nodes_count": nodes_count,
                "processing_time": processing_time
            }
            
        except Exception as e:
            error_msg = f"Vector storage failed: {str(e)}"
            state_manager.mark_stage_failed(ProcessingStage.VECTOR_STORAGE.value, error_msg)
            logger.error(f"Vector storage failed for document {state_manager.document_id}: {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "processing_time": time.time() - start_time
            }