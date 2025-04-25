#!/usr/bin/env python
# test_document_upload.py
import unittest
import os
import sys
import tempfile
import shutil
import logging
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from services.document.upload import upload_document
from core.config import AppConfig, get_config, reset_config

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_document_upload")

class TestDocumentUpload(unittest.TestCase):
    """Test cases for document upload functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temp directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_pdf_path = os.path.join(self.temp_dir, "test_document.pdf")
        
        # Create a dummy PDF file for testing
        with open(self.test_pdf_path, 'wb') as f:
            f.write(b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF')
        
        # Create a test configuration
        self.test_config = AppConfig(
            app_name="Test RAG API",
            debug=True,
            storage=AppConfig().storage.model_copy(update={"storage_dir": self.temp_dir})
        )
        
        # Reset config to ensure we're using our test config
        reset_config()
        
        # Set up mocks for dependencies
        self.setup_mocks()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temp directory
        shutil.rmtree(self.temp_dir)
        
        # Reset config
        reset_config()
    
    def setup_mocks(self):
        """Set up mock objects for dependencies."""
        # Create patch objects for all dependencies
        self.mock_patches = [
            # Mock document repository
            patch('services.document.upload.DocumentMetadataRepository'),
            # Mock PDF extraction
            patch('services.document.upload.PDFExtractor'),
            # Mock chunker
            patch('services.document.upload.Chunker'),
            # Mock embedding service
            patch('services.document.upload.EmbeddingService'),
            # Mock Raptor
            patch('services.document.upload.Raptor'),
            # Mock vector store
            patch('services.document.upload.VectorStoreAdapter')
        ]
        
        # Start all patches
        self.mocks = {}
        for p in self.mock_patches:
            mock = p.start()
            mock_name = p.target.rsplit('.', 1)[-1]  # Get the class name from the patch target
            self.mocks[mock_name] = mock
        
        # Set up return values for mocks
        
        # PDF Extractor mock
        pdf_extractor_instance = self.mocks['PDFExtractor'].return_value
        pdf_extractor_instance.extract_content.return_value = {
            "status": "success",
            "content_list": [{"type": "text", "text": "Test content", "page_idx": 0}],
            "page_count": 1
        }
        
        # Chunker mock
        chunker_instance = self.mocks['Chunker'].return_value
        chunker_instance.chunk_content.return_value = [
            {
                "id": "chunk_123",
                "content": "Test chunk content",
                "metadata": {"type": "text", "page_idx": 0}
            }
        ]
        
        # Embedding service mock
        embedding_service_instance = self.mocks['EmbeddingService'].return_value
        embedding_service_instance.generate_embeddings_dict.return_value = {
            "chunk_123": [0.1, 0.2, 0.3]
        }
        
        # Raptor mock
        raptor_instance = self.mocks['Raptor'].return_value
        raptor_instance.build_tree.return_value = {
            1: {
                "clusters_df": MagicMock(),
                "summaries_df": MagicMock()
            }
        }
        
        # Vector store mock
        vector_store_instance = self.mocks['VectorStoreAdapter'].return_value
        vector_store_instance.add_document_chunks.return_value = 1
        vector_store_instance.add_tree_nodes.return_value = 1
    
    def test_successful_upload(self):
        """Test successful document upload flow."""
        with patch('services.document.upload.get_config', return_value=self.test_config):
            result = upload_document(
                file_path=self.test_pdf_path,
                document_id="test_doc_123",
                case_id="test_case",
                metadata={"test_key": "test_value"},
                config=self.test_config
            )
            
            # Verify result structure
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["document_id"], "test_doc_123")
            self.assertEqual(result["case_id"], "test_case")
            self.assertIn("processing_time", result)
            self.assertIn("chunks_count", result)
            self.assertIn("raptor_levels", result)
            
            # Verify interactions with dependencies
            
            # Verify metadata repository calls
            doc_repo_instance = self.mocks['DocumentMetadataRepository'].return_value
            self.assertEqual(doc_repo_instance.add_document.call_count, 1)
            # Should update metadata multiple times
            self.assertGreater(doc_repo_instance.update_document.call_count, 0)
            
            # Verify PDF extraction
            pdf_extractor_instance = self.mocks['PDFExtractor'].return_value
            pdf_extractor_instance.extract_content.assert_called_once_with(self.test_pdf_path)
            
            # Verify chunking
            chunker_instance = self.mocks['Chunker'].return_value
            chunker_instance.chunk_content.assert_called_once()
            
            # Verify embedding generation
            embedding_service_instance = self.mocks['EmbeddingService'].return_value
            self.assertEqual(embedding_service_instance.generate_embeddings_dict.call_count, 2)
            
            # Verify Raptor tree building
            raptor_instance = self.mocks['Raptor'].return_value
            raptor_instance.build_tree.assert_called_once()
            
            # Verify vector storage
            vector_store_instance = self.mocks['VectorStoreAdapter'].return_value
            vector_store_instance.add_document_chunks.assert_called_once()
            vector_store_instance.add_tree_nodes.assert_called_once()
    
    def test_extraction_failure(self):
        """Test handling of extraction failure."""
        pdf_extractor_instance = self.mocks['PDFExtractor'].return_value
        pdf_extractor_instance.extract_content.return_value = {
            "status": "error",
            "message": "Failed to extract content"
        }
        
        with patch('services.document.upload.get_config', return_value=self.test_config):
            result = upload_document(
                file_path=self.test_pdf_path,
                document_id="test_doc_123",
                case_id="test_case",
                config=self.test_config
            )
            
            # Verify result structure
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["document_id"], "test_doc_123")
            self.assertEqual(result["stage"], "extraction")
            self.assertIn("error", result)
            
            # Verify update of document status in repository
            doc_repo_instance = self.mocks['DocumentMetadataRepository'].return_value
            update_call_args = doc_repo_instance.update_document.call_args[0]
            self.assertEqual(update_call_args[0], "test_doc_123")
            self.assertEqual(update_call_args[1]["status"], "failed")
    
    def test_auto_generate_document_id(self):
        """Test automatic generation of document ID when not provided."""
        with patch('services.document.upload.get_config', return_value=self.test_config):
            result = upload_document(
                file_path=self.test_pdf_path,
                case_id="test_case",
                config=self.test_config
            )
            
            # Verify document ID was generated
            self.assertIn("document_id", result)
            self.assertTrue(result["document_id"].startswith("doc_"))
            self.assertIn("test_document", result["document_id"])  # Should contain part of the filename
    
    def test_exception_handling(self):
        """Test handling of unexpected exceptions during processing."""
        # Make embedding service raise an exception
        embedding_service_instance = self.mocks['EmbeddingService'].return_value
        embedding_service_instance.generate_embeddings_dict.side_effect = Exception("Test exception")
        
        with patch('services.document.upload.get_config', return_value=self.test_config):
            result = upload_document(
                file_path=self.test_pdf_path,
                document_id="test_doc_123",
                case_id="test_case",
                config=self.test_config
            )
            
            # Verify result structure
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["document_id"], "test_doc_123")
            self.assertIn("error", result)
            self.assertEqual(result["error"], "Test exception")
            
            # Verify document status was updated
            doc_repo_instance = self.mocks['DocumentMetadataRepository'].return_value
            doc_repo_instance.update_document.assert_called_with(
                "test_doc_123", 
                {
                    "status": "failed",
                    "failure_stage": "processing",
                    "error_message": "Test exception",
                    "failure_time": unittest.mock.ANY
                }
            )

if __name__ == "__main__":
    unittest.main()