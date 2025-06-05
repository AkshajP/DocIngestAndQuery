import os
import logging
import tempfile
from typing import Dict, Any, Optional

from mineru_ingester import ingest_pdf

logger = logging.getLogger(__name__)

class PDFExtractor:
    """
    Handles PDF extraction using MinerU to extract content with spatial information.
    Enhanced with better error handling and debugging.
    """
    
    def __init__(self, language: str = 'en'):
        """
        Initialize the PDF extractor.
        
        Args:
            language: Language hint for OCR/extraction
        """
        self.language = language
    
    def extract_content(self, pdf_path: str, save_images: bool = True, output_dir: Optional[str] = None, 
                       storage_adapter=None) -> Dict[str, Any]:
        """
        Extract content from a PDF file using MinerU with enhanced error handling.
        
        Args:
            pdf_path: Path to the PDF file
            save_images: Whether to save extracted images
            output_dir: Directory to save images (if None, images saved to 'images' dir)
            storage_adapter: Storage adapter for saving images (if None, saves directly to filesystem)
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            logger.info(f"Extracting content from PDF: {pdf_path}")
            
            # Validate file before processing
            if not os.path.exists(pdf_path):
                error_msg = f"PDF file not found: {pdf_path}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            file_size = os.path.getsize(pdf_path)
            logger.info(f"PDF file size: {file_size} bytes")
            
            if file_size == 0:
                error_msg = f"PDF file is empty: {pdf_path}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            # Test file readability
            try:
                with open(pdf_path, 'rb') as f:
                    header = f.read(5)
                    if not header.startswith(b'%PDF-'):
                        error_msg = f"File is not a valid PDF: {pdf_path}"
                        logger.error(error_msg)
                        return {"status": "error", "message": error_msg}
            except Exception as e:
                error_msg = f"Cannot read PDF file: {pdf_path} - {str(e)}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            # Log MinerU configuration
            logger.info(f"Starting MinerU extraction with language: {self.language}")
            
            # Process PDF with MinerU - with detailed error capture
            try:
                extraction_result = ingest_pdf(
                    pdf_path,
                    lang=self.language,
                    dump_intermediate=True,  # Enable intermediate files for debugging
                    output_dir=output_dir
                )
                
                logger.info(f"MinerU extraction completed, result type: {type(extraction_result)}")
                
            except ImportError as e:
                error_msg = f"MinerU import error - check magic-pdf installation: {str(e)}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            except FileNotFoundError as e:
                error_msg = f"File not found during MinerU processing: {str(e)}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            except Exception as e:
                error_msg = f"MinerU processing error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return {"status": "error", "message": error_msg}
            
            # Check if extraction result is valid
            if not extraction_result:
                error_msg = f"MinerU returned empty result for PDF: {pdf_path}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            if not isinstance(extraction_result, dict):
                error_msg = f"MinerU returned invalid result type: {type(extraction_result)}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            # Validate extraction result structure
            content_list = extraction_result.get("content_list", [])
            images = extraction_result.get("images", {})
            
            logger.info(f"Extraction successful - Content items: {len(content_list)}, Images: {len(images)}")
            
            if len(content_list) == 0:
                logger.warning(f"No content extracted from PDF: {pdf_path}")
                # Don't fail completely, return empty result
                return {
                    "status": "success", 
                    "content_list": [],
                    "images": [],
                    "page_count": 0,
                    "warning": "No content extracted from PDF"
                }
            
            # Save extracted images if requested
            if save_images and images:
                try:
                    self._save_images(images, output_dir, storage_adapter)
                except Exception as e:
                    logger.warning(f"Failed to save images: {str(e)}")
                    # Don't fail the entire extraction for image save issues
            
            logger.info(f"Successfully extracted {len(content_list)} content items and {len(images)} images")
            
            return {
                "status": "success",
                "content_list": content_list,
                "images": list(images.keys()) if images else [],
                "page_count": self._count_pages(content_list),
                "image_directory": output_dir
            }
            
        except Exception as e:
            error_msg = f"Unexpected error extracting PDF content: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}
    
    def _save_images(self, images: Dict[str, bytes], output_dir: Optional[str] = None, 
                    storage_adapter=None) -> None:
        """
        Save extracted images to disk or using storage adapter.
        
        Args:
            images: Dictionary of image name to image data
            output_dir: Directory to save images (default: 'images')
            storage_adapter: Storage adapter for saving images (if None, saves directly to filesystem)
        """
        # Determine output directory
        if not output_dir:
            output_dir = "images"
        
        # Save each image
        for image_name, image_data in images.items():
            image_path = os.path.join(output_dir, image_name)
            
            try:
                if storage_adapter:
                    # Use storage adapter if provided
                    # Ensure parent directory exists
                    parent_dir = os.path.dirname(image_path)
                    if parent_dir:
                        storage_adapter.create_directory(parent_dir)
                    
                    # Write image data
                    storage_adapter.write_file(image_data, image_path)
                    logger.debug(f"Saved image to {image_path} using storage adapter")
                else:
                    # Direct filesystem access if no adapter provided
                    # Create subdirectories if needed
                    parent_dir = os.path.dirname(image_path)
                    if parent_dir:
                        os.makedirs(parent_dir, exist_ok=True)
                    
                    # Write image data
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                    logger.debug(f"Saved image to {image_path}")
                
            except Exception as e:
                logger.error(f"Error saving image {image_name}: {str(e)}")
    
    def _count_pages(self, content_list: list) -> int:
        """
        Count the number of unique pages in the content list.
        
        Args:
            content_list: List of content items
            
        Returns:
            Number of unique pages
        """
        if not content_list:
            return 0
            
        pages = set()
        for item in content_list:
            if "page_idx" in item:
                pages.add(item["page_idx"])
        
        return len(pages)