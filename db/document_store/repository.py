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
    
    def get_document(self, document_id: str, user_id: Optional[str] = None, case_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get document metadata with access control.
        
        Args:
            document_id: Document ID
            user_id: Optional user ID for access control
            case_id: Optional case ID for verification
            
        Returns:
            Document metadata dictionary or None if not found or no access
        """
        with self.metadata_lock:
            document = self.registry["documents"].get(document_id)
            
            # If document not found, return None
            if not document:
                return None
                
            # If case_id provided, verify document belongs to that case
            if case_id and document.get("case_id") != case_id:
                return None
                
            # If user_id provided, verify they have access to the document's case
            # This would check user_case_access in a complete implementation
            
            return document
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the registry.
        Necessarily an admin function, since it bypasses case filter.
        
        Returns:
            List of document metadata dictionaries
        """
        with self.metadata_lock:
            return list(self.registry["documents"].values())
    
    def list_documents_by_case(self, case_id: str) -> List[Dict[str, Any]]:
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
    def add_document_with_processing_state(
        self, 
        document_metadata: Dict[str, Any],
        initial_stage: str = "upload"
    ) -> bool:
        """
        Add a document to the registry with initial processing state.
        
        Args:
            document_metadata: Document metadata dictionary
            initial_stage: Initial processing stage
            
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
                
                # Initialize processing state
                processing_state = {
                    "current_stage": initial_stage,
                    "completed_stages": [],
                    "stage_data_paths": {},
                    "stage_completion_times": {},
                    "stage_error_details": {},
                    "retry_counts": {},
                    "last_updated": datetime.now().isoformat()
                }
                
                # Add processing state to metadata
                enhanced_metadata = {
                    **document_metadata,
                    "processing_state": processing_state
                }
                
                self.registry["documents"][document_id] = enhanced_metadata
                
                # Save to disk
                return self._save_registry()
            except Exception as e:
                logger.error(f"Error adding document to registry: {str(e)}")
                return False
    
    def update_processing_state(
        self, 
        document_id: str, 
        processing_state: Dict[str, Any]
    ) -> bool:
        """
        Update the processing state for a document.
        
        Args:
            document_id: Document ID
            processing_state: Updated processing state
            
        Returns:
            True if successful, False otherwise
        """
        with self.metadata_lock:
            try:
                if document_id not in self.registry["documents"]:
                    logger.warning(f"Document {document_id} not found in registry for processing state update")
                    return False
                
                # Update processing state
                current_metadata = self.registry["documents"][document_id]
                current_metadata["processing_state"] = processing_state
                current_metadata["processing_state"]["last_updated"] = datetime.now().isoformat()
                
                # Save to disk
                return self._save_registry()
            except Exception as e:
                logger.error(f"Error updating processing state for document {document_id}: {str(e)}")
                return False
    
    def get_documents_by_processing_stage(self, stage: str) -> List[Dict[str, Any]]:
        """
        Get all documents currently at a specific processing stage.
        
        Args:
            stage: Processing stage to filter by
            
        Returns:
            List of document metadata dictionaries
        """
        with self.metadata_lock:
            return [
                doc for doc in self.registry["documents"].values()
                if doc.get("processing_state", {}).get("current_stage") == stage
            ]
    def get_failed_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents that have failed at any stage.
        
        Returns:
            List of document metadata dictionaries with error details
        """
        with self.metadata_lock:
            failed_docs = []
            for doc in self.registry["documents"].values():
                processing_state = doc.get("processing_state", {})
                stage_errors = processing_state.get("stage_error_details", {})
                
                if stage_errors or doc.get("status") == "failed":
                    # Add error summary to document
                    doc_with_errors = doc.copy()
                    doc_with_errors["error_summary"] = {
                        "failed_stages": list(stage_errors.keys()),
                        "total_retry_attempts": sum(processing_state.get("retry_counts", {}).values()),
                        "last_error": max(stage_errors.values(), key=lambda x: x.get("timestamp", "")) if stage_errors else None
                    }
                    failed_docs.append(doc_with_errors)
            
            return failed_docs
    
    def get_processing_statistics_detailed(self) -> Dict[str, Any]:
        """
        Get detailed statistics about document processing including stage information.
        
        Returns:
            Dictionary of detailed statistics
        """
        with self.metadata_lock:
            total_documents = len(self.registry["documents"])
            
            # Initialize counters
            by_status = {}
            by_stage = {}
            stage_completion_times = {}
            stage_error_counts = {}
            retry_statistics = {}
            
            # Analyze each document
            for doc in self.registry["documents"].values():
                # Status statistics
                status = doc.get("status", "unknown")
                by_status[status] = by_status.get(status, 0) + 1
                
                # Processing state analysis
                processing_state = doc.get("processing_state", {})
                current_stage = processing_state.get("current_stage", "unknown")
                by_stage[current_stage] = by_stage.get(current_stage, 0) + 1
                
                # Stage completion times
                completion_times = processing_state.get("stage_completion_times", {})
                for stage, completion_time in completion_times.items():
                    if stage not in stage_completion_times:
                        stage_completion_times[stage] = []
                    try:
                        # Calculate time from start to completion (simplified)
                        stage_completion_times[stage].append(completion_time)
                    except:
                        pass
                
                # Stage error counts
                stage_errors = processing_state.get("stage_error_details", {})
                for stage in stage_errors.keys():
                    stage_error_counts[stage] = stage_error_counts.get(stage, 0) + 1
                
                # Retry statistics
                retry_counts = processing_state.get("retry_counts", {})
                for stage, count in retry_counts.items():
                    if stage not in retry_statistics:
                        retry_statistics[stage] = []
                    retry_statistics[stage].append(count)
            
            # Calculate averages for retry statistics
            retry_summary = {}
            for stage, counts in retry_statistics.items():
                retry_summary[stage] = {
                    "total_retries": sum(counts),
                    "avg_retries_per_doc": sum(counts) / len(counts) if counts else 0,
                    "max_retries": max(counts) if counts else 0,
                    "documents_with_retries": len(counts)
                }
            
            return {
                "total_documents": total_documents,
                "by_status": by_status,
                "by_current_stage": by_stage,
                "stage_completion_counts": {stage: len(times) for stage, times in stage_completion_times.items()},
                "stage_error_counts": stage_error_counts,
                "retry_statistics": retry_summary,
                "last_updated": self.registry.get("last_updated")
            }
    
    def get_documents_needing_retry(
        self, 
        max_retry_count: int = 3,
        min_time_since_failure: int = 300  # 5 minutes
    ) -> List[Dict[str, Any]]:
        """
        Get documents that failed but are eligible for retry.
        
        Args:
            max_retry_count: Maximum number of retries allowed
            min_time_since_failure: Minimum seconds since last failure before retry
            
        Returns:
            List of documents eligible for retry
        """
        with self.metadata_lock:
            eligible_docs = []
            current_time = datetime.now()
            
            for doc in self.registry["documents"].values():
                # Check if document has failed
                if doc.get("status") != "failed":
                    continue
                
                processing_state = doc.get("processing_state", {})
                stage_errors = processing_state.get("stage_error_details", {})
                retry_counts = processing_state.get("retry_counts", {})
                
                # Check if any stage has failed
                if not stage_errors:
                    continue
                
                # Check retry limits
                total_retries = sum(retry_counts.values())
                if total_retries >= max_retry_count:
                    continue
                
                # Check time since last failure
                latest_error = max(stage_errors.values(), key=lambda x: x.get("timestamp", ""))
                try:
                    error_time = datetime.fromisoformat(latest_error["timestamp"])
                    time_diff = (current_time - error_time).total_seconds()
                    
                    if time_diff >= min_time_since_failure:
                        eligible_docs.append({
                            **doc,
                            "retry_eligibility": {
                                "total_retries": total_retries,
                                "time_since_failure": time_diff,
                                "last_failed_stage": max(stage_errors.keys()),
                                "recommended_retry_stage": max(stage_errors.keys())
                            }
                        })
                except:
                    # If timestamp parsing fails, include in eligible list
                    eligible_docs.append(doc)
            
            return eligible_docs
    
    def cleanup_old_processing_data(self, days_old: int = 30) -> int:
        """
        Clean up processing data for old documents.
        
        Args:
            days_old: Remove processing data older than this many days
            
        Returns:
            Number of documents cleaned up
        """
        with self.metadata_lock:
            cleaned_count = 0
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            for doc_id, doc in self.registry["documents"].items():
                processing_state = doc.get("processing_state", {})
                last_updated = processing_state.get("last_updated")
                
                if last_updated:
                    try:
                        update_time = datetime.fromisoformat(last_updated).timestamp()
                        if update_time < cutoff_date:
                            # Clean up stage data paths and error details
                            processing_state["stage_data_paths"] = {}
                            processing_state["stage_error_details"] = {}
                            cleaned_count += 1
                    except:
                        pass
            
            if cleaned_count > 0:
                self._save_registry()
                logger.info(f"Cleaned up processing data for {cleaned_count} old documents")
            
            return cleaned_count
    
    def get_document_with_processing_history(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document with complete processing history and timeline.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document with processing timeline or None if not found
        """
        with self.metadata_lock:
            document = self.registry["documents"].get(document_id)
            
            if not document:
                return None
            
            processing_state = document.get("processing_state", {})
            
            # Build processing timeline
            timeline = []
            
            # Add completion times
            completion_times = processing_state.get("stage_completion_times", {})
            for stage, timestamp in completion_times.items():
                timeline.append({
                    "event": "stage_completed",
                    "stage": stage,
                    "timestamp": timestamp,
                    "details": f"Stage {stage} completed successfully"
                })
            
            # Add error events
            stage_errors = processing_state.get("stage_error_details", {})
            for stage, error_info in stage_errors.items():
                timeline.append({
                    "event": "stage_failed",
                    "stage": stage,
                    "timestamp": error_info.get("timestamp"),
                    "details": error_info.get("error", "Unknown error")
                })
            
            # Sort timeline by timestamp
            timeline.sort(key=lambda x: x.get("timestamp", ""))
            
            # Add timeline to document
            enhanced_document = document.copy()
            enhanced_document["processing_timeline"] = timeline
            enhanced_document["processing_summary"] = {
                "current_stage": processing_state.get("current_stage"),
                "completed_stages_count": len(processing_state.get("completed_stages", [])),
                "failed_stages_count": len(stage_errors),
                "total_retry_attempts": sum(processing_state.get("retry_counts", {}).values()),
                "last_activity": processing_state.get("last_updated")
            }
            
            return enhanced_document