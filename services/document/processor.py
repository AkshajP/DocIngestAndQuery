import os
import time
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.config import get_config
from db.document_store.repository import DocumentMetadataRepository
from db.vector_store.adapter import VectorStoreAdapter
from services.document.storage import StorageAdapter, LocalStorageAdapter, S3StorageAdapter
from services.document.chunker import Chunker
from services.pdf.extractor import PDFExtractor
from services.pdf.highlighter import PDFHighlighter
from services.ml.embeddings import EmbeddingService
from services.retrieval.raptor import Raptor

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Main orchestration class for document processing.
    Coordinates the document processing pipeline from upload to storage.
    """
    
    def __init__(self, config=None):
        """
        Initialize the document processor.
        
        Args:
            config: Configuration object (if None, will load from global config)
        """
        self.config = config or get_config()
        
        # Initialize storage adapter based on configuration
        self.storage_adapter = self._initialize_storage_adapter()
        
        # Initialize document metadata repository
        self.metadata_repo = DocumentMetadataRepository()
        
        # Initialize vector store adapter
        self.vector_store = VectorStoreAdapter()
        
        # Initialize document processing components
        self.pdf_extractor = PDFExtractor(language=self.config.processing.language)
        self.chunker = Chunker(
            max_chunk_size=self.config.processing.chunk_size,
            min_chunk_size=200,
            overlap_size=self.config.processing.chunk_overlap
        )
        self.embedding_service = EmbeddingService(
            model_name=self.config.ollama.embed_model,
            base_url=self.config.ollama.base_url
        )
        
        # Initialize PDF highlighter for thumbnails
        self.pdf_highlighter = PDFHighlighter(storage_dir=self.config.storage.storage_dir)
        
        # Initialize RAPTOR for tree building
        self.raptor = Raptor(
            ollama_base_url=self.config.ollama.base_url,
            ollama_model=self.config.ollama.model,
            ollama_embed_model=self.config.ollama.embed_model,
            max_tree_levels=self.config.processing.max_tree_levels
        )
        
        logger.info("Document processor initialized successfully")
    
    def _initialize_storage_adapter(self) -> StorageAdapter:
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
    
    def process_document(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        case_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a document through the entire pipeline.
        
        Args:
            file_path: Path to the document file
            document_id: Optional custom ID (generated if not provided)
            case_id: Case ID for document grouping (required)
            metadata: Optional metadata about the document
            
        Returns:
            Dictionary with processing status and information
        """
        start_time = time.time()
        
        # Generate document ID if not provided
        if not document_id:
            timestamp = int(time.time())
            base_name = os.path.basename(file_path)
            safe_name = ''.join(c for c in base_name.split('.')[0].replace(' ', '_') 
                              if c.isalnum() or c == '_')
            document_id = f"doc_{timestamp}_{safe_name}"
        
        # Initialize metadata
        doc_metadata = {
            "document_id": document_id,
            "case_id": case_id,
            "original_filename": os.path.basename(file_path),
            "original_file_path": file_path,
            "file_type": os.path.splitext(file_path)[1].lower()[1:],
            "processing_start_time": datetime.now().isoformat(),
            "status": "processing",
            "user_metadata": metadata or {}
        }
        
        # Save initial metadata
        self.metadata_repo.add_document(doc_metadata)
        
        try:
            # Step 1: Extract content from document
            extraction_result = self._extract_document_content(file_path, document_id, case_id)
            if extraction_result["status"] != "success":
                return self._handle_processing_failure(document_id, "extraction", extraction_result["message"])
            
            # Update metadata with extraction info
            self.metadata_repo.update_document(document_id, {
                "page_count": extraction_result.get("page_count", 0),
                "extraction_time": extraction_result.get("processing_time", 0)
            })
            
            # Step 2: Chunk the extracted content
            chunking_result = self._chunk_document_content(
                extraction_result["content_list"], 
                document_id, 
                case_id
            )
            if chunking_result["status"] != "success":
                return self._handle_processing_failure(document_id, "chunking", chunking_result["message"])
            
            # Update metadata with chunking info
            self.metadata_repo.update_document(document_id, {
                "chunks_count": chunking_result.get("chunks_count", 0),
                "content_types": chunking_result.get("content_types", {}),
                "chunking_time": chunking_result.get("processing_time", 0)
            })
            
            # Step 3: Generate embeddings for chunks
            embedding_result = self._generate_embeddings(
                chunking_result["chunks"], 
                document_id, 
                case_id
            )
            if embedding_result["status"] != "success":
                return self._handle_processing_failure(document_id, "embedding", embedding_result["message"])
            
            # Update metadata with embedding info
            self.metadata_repo.update_document(document_id, {
                "embedding_time": embedding_result.get("processing_time", 0)
            })
            
            # Step 4: Build RAPTOR tree
            tree_result = self._build_raptor_tree(
                chunking_result["chunks"],
                embedding_result["embeddings"],
                document_id,
                case_id
            )
            if tree_result["status"] != "success":
                return self._handle_processing_failure(document_id, "tree_building", tree_result["message"])
            
            # Step 5: Store document chunks and tree in vector database
            storage_result = self._store_in_vector_database(
                document_id,
                case_id,
                chunking_result["chunks"],
                embedding_result["embeddings"],
                tree_result["tree_data"],
                tree_result["tree_embeddings"]
            )
            if storage_result["status"] != "success":
                return self._handle_processing_failure(document_id, "vector_storage", storage_result["message"])
            
            # Calculate total processing time
            processing_time = time.time() - start_time
            
            # Update document metadata with success status
            self.metadata_repo.update_document(document_id, {
                "status": "processed",
                "processing_time": processing_time,
                "raptor_levels": tree_result.get("raptor_levels", []),
                "processing_date": datetime.now().isoformat()
            })
            
            logger.info(f"Document {document_id} processed successfully in {processing_time:.2f} seconds")
            
            return {
                "status": "success",
                "document_id": document_id,
                "case_id": case_id,
                "processing_time": processing_time,
                "chunks_count": chunking_result.get("chunks_count", 0),
                "raptor_levels": tree_result.get("raptor_levels", [])
            }
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            return self._handle_processing_failure(document_id, "processing", str(e))
    
    def _extract_document_content(
        self, 
        file_path: str, 
        document_id: str,
        case_id: str
    ) -> Dict[str, Any]:
        """
        Extract content from the document.
        
        Args:
            file_path: Path to the document file
            document_id: Document ID
            case_id: Case ID
            
        Returns:
            Dictionary with extraction results
        """
        start_time = time.time()
        
        # Check file type
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            # Extract PDF content
            extraction_result = self.pdf_extractor.extract_content(file_path)
            
            if extraction_result["status"] != "success":
                return {
                    "status": "error",
                    "message": extraction_result.get("message", "PDF extraction failed")
                }
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "content_list": extraction_result["content_list"],
                "page_count": extraction_result.get("page_count", 0),
                "processing_time": processing_time
            }
        else:
            return {
                "status": "error",
                "message": f"Unsupported file type: {file_ext}"
            }
    
    def _chunk_document_content(
        self, 
        content_list: List[Dict[str, Any]], 
        document_id: str,
        case_id: str
    ) -> Dict[str, Any]:
        """
        Chunk the document content.
        
        Args:
            content_list: List of content items from extraction
            document_id: Document ID
            case_id: Case ID
            
        Returns:
            Dictionary with chunking results
        """
        start_time = time.time()
        
        # Process content using the chunker
        chunks = self.chunker.chunk_content(content_list)
        
        if not chunks:
            return {
                "status": "error",
                "message": "No chunks created from document content"
            }
        
        # Count content types
        content_types = {}
        for chunk in chunks:
            chunk_type = chunk.get("metadata", {}).get("type", "unknown")
            if chunk_type not in content_types:
                content_types[chunk_type] = 0
            content_types[chunk_type] += 1
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "chunks": chunks,
            "chunks_count": len(chunks),
            "content_types": content_types,
            "processing_time": processing_time
        }
    
    def _generate_embeddings(
        self, 
        chunks: List[Dict[str, Any]], 
        document_id: str,
        case_id: str
    ) -> Dict[str, Any]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of document chunks
            document_id: Document ID
            case_id: Case ID
            
        Returns:
            Dictionary with embedding results
        """
        start_time = time.time()
        
        # Prepare texts dictionary with chunk IDs and contents
        texts_dict = {chunk["id"]: chunk["content"] for chunk in chunks}
        
        # Generate embeddings
        embeddings = self.embedding_service.generate_embeddings_dict(
            texts_dict, 
            show_progress=True
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "embeddings": embeddings,
            "processing_time": processing_time
        }
    
    def _build_raptor_tree(
        self, 
        chunks: List[Dict[str, Any]], 
        embeddings: Dict[str, List[float]],
        document_id: str,
        case_id: str
    ) -> Dict[str, Any]:
        """
        Build RAPTOR hierarchical tree structure.
        
        Args:
            chunks: List of document chunks
            embeddings: Dictionary mapping chunk IDs to embeddings
            document_id: Document ID
            case_id: Case ID
            
        Returns:
            Dictionary with tree building results
        """
        start_time = time.time()
        
        # Format chunks for RAPTOR
        chunk_texts = [chunk["content"] for chunk in chunks]
        chunk_ids = [chunk["id"] for chunk in chunks]
        
        # Pass to RAPTOR for tree building
        try:
            tree_data = self.raptor.build_tree(chunk_texts, chunk_ids, embeddings)
            
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
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "tree_data": tree_data,
                "tree_embeddings": tree_embeddings,
                "raptor_levels": sorted(list(tree_data.keys())),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error building RAPTOR tree: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to build RAPTOR tree: {str(e)}"
            }
    
    def _store_in_vector_database(
        self,
        document_id: str,
        case_id: str,
        chunks: List[Dict[str, Any]],
        chunk_embeddings: Dict[str, List[float]],
        tree_data: Dict[int, Dict[str, Any]],
        tree_embeddings: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Store document chunks and tree nodes in vector database.
        
        Args:
            document_id: Document ID
            case_id: Case ID
            chunks: List of document chunks
            chunk_embeddings: Dictionary mapping chunk IDs to embeddings
            tree_data: RAPTOR tree data
            tree_embeddings: Dictionary mapping tree node IDs to embeddings
            
        Returns:
            Dictionary with storage results
        """
        start_time = time.time()
        
        # Store original chunks
        chunks_count = self.vector_store.add_document_chunks(
            document_id, 
            case_id, 
            chunks, 
            chunk_embeddings
        )
        
        if chunks_count == 0:
            return {
                "status": "error",
                "message": "Failed to store document chunks in vector database"
            }
        
        # Store tree nodes
        nodes_count = self.vector_store.add_tree_nodes(
            document_id, 
            case_id, 
            tree_data, 
            tree_embeddings
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "chunks_count": chunks_count,
            "nodes_count": nodes_count,
            "processing_time": processing_time
        }
    
    def _handle_processing_failure(
        self, 
        document_id: str, 
        stage: str, 
        error_message: str
    ) -> Dict[str, Any]:
        """
        Handle document processing failure.
        
        Args:
            document_id: Document ID
            stage: Processing stage that failed
            error_message: Error message
            
        Returns:
            Error response dictionary
        """
        failure_metadata = {
            "status": "failed",
            "failure_stage": stage,
            "error_message": error_message,
            "failure_time": datetime.now().isoformat()
        }
        
        # Update document metadata
        self.metadata_repo.update_document(document_id, failure_metadata)
        
        logger.error(f"Document {document_id} processing failed at {stage}: {error_message}")
        
        return {
            "status": "error",
            "document_id": document_id,
            "stage": stage,
            "error": error_message
        }
    
    def delete_document(self, document_id: str, case_id: str) -> bool:
        """
        Delete a document and all its data.
        
        Args:
            document_id: Document ID
            case_id: Case ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete from vector database
            vector_deleted = self.vector_store.delete_document(document_id, case_id)
            
            # Delete from document registry
            metadata_deleted = self.metadata_repo.delete_document(document_id)
            
            # Log result
            if vector_deleted and metadata_deleted:
                logger.info(f"Document {document_id} deleted successfully")
                return True
            else:
                logger.warning(f"Partial deletion for document {document_id}: vector_deleted={vector_deleted}, metadata_deleted={metadata_deleted}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the registry.
        
        Returns:
            List of document metadata dictionaries
        """
        return self.metadata_repo.list_documents()
    
    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document metadata dictionary or None if not found
        """
        return self.metadata_repo.get_document(document_id)
    
    def find_documents_by_case(self, case_id: str) -> List[Dict[str, Any]]:
        """
        Find documents belonging to a specific case.
        
        Args:
            case_id: Case ID
            
        Returns:
            List of document metadata dictionaries
        """
        return self.metadata_repo.find_documents_by_case(case_id)