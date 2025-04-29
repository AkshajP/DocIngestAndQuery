import logging
import threading
import queue
import time
import uuid
from typing import Dict, Any, Optional, Callable

from services.document.processor import DocumentProcessor

logger = logging.getLogger(__name__)

class JobStatus:
    """Job status constants"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DocumentJob:
    """
    Represents a document processing job.
    """
    
    def __init__(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        case_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """
        Initialize a document processing job.
        
        Args:
            file_path: Path to the document file
            document_id: Optional custom ID (generated if not provided)
            case_id: Case ID for document grouping
            metadata: Optional metadata about the document
            callback: Optional callback function to call when job completes
        """
        self.job_id = f"job_{uuid.uuid4().hex[:10]}"
        self.file_path = file_path
        self.document_id = document_id
        self.case_id = case_id
        self.metadata = metadata or {}
        self.callback = callback
        self.status = JobStatus.PENDING
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.progress = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary representation"""
        return {
            "job_id": self.job_id,
            "document_id": self.document_id,
            "case_id": self.case_id,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error
        }

class DocumentJobProcessor:
    """
    Processes document jobs in background threads.
    """
    
    def __init__(self, max_workers: int = 2):
        """
        Initialize the document job processor.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.job_queue = queue.Queue()
        self.jobs = {}  # job_id -> DocumentJob
        self.max_workers = max_workers
        self.workers = []
        self.stop_event = threading.Event()
        self.document_processor = DocumentProcessor()
        self._start_workers()
    
    def _start_workers(self) -> None:
        """Start worker threads"""
        for _ in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        logger.info(f"Started {self.max_workers} document job worker threads")
    
    def _worker_loop(self) -> None:
        """Worker thread main loop"""
        while not self.stop_event.is_set():
            try:
                # Get job from queue with timeout to check stop_event periodically
                job = self.job_queue.get(timeout=1.0)
                
                # Process job
                self._process_job(job)
                
                # Mark task as done
                self.job_queue.task_done()
                
            except queue.Empty:
                # No jobs in queue, continue checking stop_event
                continue
            except Exception as e:
                logger.error(f"Error in worker thread: {str(e)}")
    
    def _process_job(self, job: DocumentJob) -> None:
        """
        Process a document job.
        
        Args:
            job: Document job to process
        """
        # Update job status
        job.status = JobStatus.PROCESSING
        job.started_at = time.time()
        job.progress = 10
        
        try:
            # Process document
            result = self.document_processor.process_document(
                file_path=job.file_path,
                document_id=job.document_id,
                case_id=job.case_id,
                metadata=job.metadata
            )
            
            # Update job with result
            job.progress = 100
            
            if result["status"] == "success":
                job.status = JobStatus.COMPLETED
                job.result = result
                logger.info(f"Job {job.job_id} completed successfully")
            else:
                job.status = JobStatus.FAILED
                job.error = result.get("error", "Unknown error")
                logger.error(f"Job {job.job_id} failed: {job.error}")
            
        except Exception as e:
            # Handle exceptions
            job.status = JobStatus.FAILED
            job.error = str(e)
            logger.error(f"Error processing job {job.job_id}: {str(e)}")
        
        finally:
            # Always mark job as completed
            job.completed_at = time.time()
            
            # Call callback if provided
            if job.callback:
                try:
                    job.callback(job.job_id, job.to_dict())
                except Exception as e:
                    logger.error(f"Error in job callback: {str(e)}")
    
    def submit_job(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        case_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> str:
        """
        Submit a document processing job.
        
        Args:
            file_path: Path to the document file
            document_id: Optional custom ID (generated if not provided)
            case_id: Case ID for document grouping
            metadata: Optional metadata about the document
            callback: Optional callback function
            
        Returns:
            Job ID
        """
        # Create job
        job = DocumentJob(
            file_path=file_path,
            document_id=document_id,
            case_id=case_id,
            metadata=metadata,
            callback=callback
        )
        
        # Store job
        self.jobs[job.job_id] = job
        
        # Add to queue
        self.job_queue.put(job)
        
        logger.info(f"Submitted job {job.job_id} for document {file_path}")
        
        return job.job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status dictionary or None if not found
        """
        job = self.jobs.get(job_id)
        if job:
            return job.to_dict()
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if job was cancelled, False otherwise
        """
        job = self.jobs.get(job_id)
        if job and job.status == JobStatus.PENDING:
            # Remove from queue (not easily possible with Python's queue)
            # Mark as cancelled
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            return True
        return False
    
    def shutdown(self) -> None:
        """Shutdown the job processor"""
        logger.info("Shutting down document job processor")
        self.stop_event.set()
        
        # Wait for all workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        logger.info("Document job processor shutdown complete")

# Global instance
_job_processor = None

def get_job_processor() -> DocumentJobProcessor:
    """Get the global job processor instance"""
    global _job_processor
    if _job_processor is None:
        _job_processor = DocumentJobProcessor()
    return _job_processor