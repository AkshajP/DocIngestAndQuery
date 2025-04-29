import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import threading

logger = logging.getLogger(__name__)

class DocumentMetadataRepository:
    """
    Repository for document metadata storage and retrieval.
    Stores document metadata in a JSON file for persistence.
    """
    
    def __init__(self, storage_path: str = "document_store/registry.json"):
        """
        Initialize the document metadata repository.
        
        Args:
            storage_path: Path to the metadata storage file
        """
        self.storage_path = storage_path
        self.metadata_lock = threading.RLock()  # Reentrant lock for thread safety
        self._ensure_storage_directory()
        self._load_registry()
    
    def _ensure_storage_directory(self):
        """Ensure the storage directory exists"""
        directory = os.path.dirname(self.storage_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created storage directory: {directory}")
    
    def _load_registry(self):
        """Load the document registry from disk"""
        with self.metadata_lock:
            if os.path.exists(self.storage_path):
                try:
                    with open(self.storage_path, 'r', encoding='utf-8') as f:
                        self.registry = json.load(f)
                        logger.info(f"Loaded document registry from {self.storage_path}")
                except Exception as e:
                    logger.error(f"Error loading document registry: {str(e)}")
                    self._initialize_empty_registry()
            else:
                logger.warning(f"Document registry not found at {self.storage_path}, creating new")
                self._initialize_empty_registry()
    
    def _initialize_empty_registry(self):
        """Initialize an empty document registry"""
        self.registry = {
            "documents": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_registry(self):
        """Save the document registry to disk"""
        with self.metadata_lock:
            try:
                # Update last updated timestamp
                self.registry["last_updated"] = datetime.now().isoformat()
                
                # Create directory if it doesn't exist
                self._ensure_storage_directory()
                
                # Write to file
                with open(self.storage_path, 'w', encoding='utf-8') as f:
                    json.dump(self.registry, f, indent=2)
                
                logger.info(f"Saved document registry to {self.storage_path}")
                return True
            except Exception as e:
                logger.error(f"Error saving document registry: {str(e)}")
                return False
    
    def add_document(self, document_metadata: Dict[str, Any]) -> bool:
        """
        Add a document to the registry.
        
        Args:
            document_metadata: Document metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        with self.metadata_lock:
            try:
                document_id = document_metadata.get("document_id")
                if not document_id:
                    logger.error("Document ID is required for adding to registry")
                    return False
                
                # Ensure case_id exists in metadata
                if "case_id" not in document_metadata:
                    logger.error("Case ID is required for adding to registry")
                    return False
                
                # Ensure document_id exists in metadata
                self.registry["documents"][document_id] = document_metadata
                
                # Save to disk
                return self._save_registry()
            except Exception as e:
                logger.error(f"Error adding document to registry: {str(e)}")
                return False
    
    def update_document(self, document_id: str, metadata_updates: Dict[str, Any]) -> bool:
        """
        Update document metadata.
        
        Args:
            document_id: Document ID
            metadata_updates: Dictionary of metadata updates
            
        Returns:
            True if successful, False otherwise
        """
        with self.metadata_lock:
            try:
                if document_id not in self.registry["documents"]:
                    logger.warning(f"Document {document_id} not found in registry for update")
                    return False
                
                # Update document metadata
                current_metadata = self.registry["documents"][document_id]
                updated_metadata = {**current_metadata, **metadata_updates}
                
                # Update processing time if processing has completed
                if ("status" in metadata_updates and 
                    metadata_updates["status"] == "processed" and
                    "processing_start_time" in current_metadata):
                    
                    # Calculate processing time if not provided
                    if "processing_time" not in metadata_updates:
                        start_time = datetime.fromisoformat(current_metadata["processing_start_time"])
                        end_time = datetime.now()
                        processing_time = (end_time - start_time).total_seconds()
                        updated_metadata["processing_time"] = processing_time
                
                # Update document in registry
                self.registry["documents"][document_id] = updated_metadata
                
                # Save to disk
                return self._save_registry()
            except Exception as e:
                logger.error(f"Error updating document in registry: {str(e)}")
                return False
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the registry.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        with self.metadata_lock:
            try:
                if document_id not in self.registry["documents"]:
                    logger.warning(f"Document {document_id} not found in registry for deletion")
                    return False
                
                # Delete document from registry
                del self.registry["documents"][document_id]
                
                # Save to disk
                return self._save_registry()
            except Exception as e:
                logger.error(f"Error deleting document from registry: {str(e)}")
                return False
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document metadata dictionary or None if not found
        """
        with self.metadata_lock:
            return self.registry["documents"].get(document_id)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the registry.
        
        Returns:
            List of document metadata dictionaries
        """
        with self.metadata_lock:
            return list(self.registry["documents"].values())
    
    def find_documents_by_case(self, case_id: str) -> List[Dict[str, Any]]:
        """
        Find documents belonging to a specific case.
        
        Args:
            case_id: Case ID to search for
            
        Returns:
            List of document metadata dictionaries
        """
        with self.metadata_lock:
            return [
                doc for doc in self.registry["documents"].values()
                if doc.get("case_id") == case_id
            ]
    
    def find_document_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Find a document by its original filename.
        
        Args:
            filename: Original filename to search for
            
        Returns:
            Document metadata dictionary or None if not found
        """
        with self.metadata_lock:
            for doc in self.registry["documents"].values():
                if doc.get("original_filename") == filename:
                    return doc
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the documents in the registry.
        
        Returns:
            Dictionary of statistics
        """
        with self.metadata_lock:
            total_documents = len(self.registry["documents"])
            processed_documents = len([
                doc for doc in self.registry["documents"].values() 
                if doc.get("status") == "processed"
            ])
            failed_documents = len([
                doc for doc in self.registry["documents"].values() 
                if doc.get("status") == "failed"
            ])
            
            # Count by case
            cases = {}
            for doc in self.registry["documents"].values():
                case_id = doc.get("case_id", "unknown")
                if case_id not in cases:
                    cases[case_id] = 0
                cases[case_id] += 1
            
            # Calculate total pages and chunks
            total_pages = sum(
                doc.get("page_count", 0) 
                for doc in self.registry["documents"].values()
            )
            
            total_chunks = sum(
                doc.get("chunks_count", 0) 
                for doc in self.registry["documents"].values()
            )
            
            return {
                "total_documents": total_documents,
                "processed_documents": processed_documents,
                "failed_documents": failed_documents,
                "total_pages": total_pages,
                "total_chunks": total_chunks,
                "documents_by_case": cases,
                "last_updated": self.registry.get("last_updated")
            }