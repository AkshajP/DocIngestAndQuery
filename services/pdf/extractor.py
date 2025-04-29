import os
import logging
import tempfile
from typing import Dict, Any, Optional

from mineru_ingester import ingest_pdf

logger = logging.getLogger(__name__)

class PDFExtractor:
    """
    Handles PDF extraction using MinerU to extract content with spatial information.
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
        Extract content from a PDF file using MinerU.
        
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
            
            # Process PDF with MinerU
            extraction_result = ingest_pdf(
                pdf_path,
                lang=self.language
            )
            
            if not extraction_result:
                logger.error(f"Failed to extract content from PDF: {pdf_path}")
                return {"status": "error", "message": "Extraction failed"}
            
            content_list = extraction_result.get("content_list", [])
            images = extraction_result.get("images", {})
            
            # Save extracted images if requested
            if save_images and images:
                self._save_images(images, output_dir, storage_adapter)
            
            logger.info(f"Successfully extracted {len(content_list)} content items and {len(images)} images")
            
            return {
                "status": "success",
                "content_list": content_list,
                "images": list(images.keys()) if images else [],
                "page_count": self._count_pages(content_list),
                "image_directory": output_dir
            }
            
        except Exception as e:
            logger.error(f"Error extracting PDF content: {str(e)}")
            return {"status": "error", "message": str(e)}
    
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