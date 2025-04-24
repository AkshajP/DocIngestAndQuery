import os
import fitz  # PyMuPDF
import base64
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger('pdf_utils')

class PDFHighlighter:
    """
    Utility class for rendering highlighted regions from PDFs as thumbnails
    """
    
    def __init__(self, storage_dir="document_store", highlight_color=(0.3, 0.4, 0, 0.4)):
        """
        Initialize the PDF highlighter.
        
        Args:
            storage_dir: Base directory for stored documents
        """
        self.storage_dir = storage_dir
        self.highlight_color = highlight_color
    
    def get_thumbnail_for_bbox(self, document_id, page_idx, bbox, 
                               highlight_color=None,  # yellow with 30% opacity
                               zoom=2.0, padding=20):
        """
        Generate a thumbnail image of a specific region in a PDF with highlighting.
        
        Args:
            document_id: Document ID
            page_idx: Page index (0-based)
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            highlight_color: RGBA tuple for highlight color (0-1 range)
            zoom: Zoom factor for higher resolution
            padding: Padding around the bbox in pixels
            
        Returns:
            Base64 encoded image string for HTML display or None if error
        """
        try:
            if highlight_color is None:
                highlight_color = self.highlight_color
            # Find the path to the original PDF
            pdf_path = self._find_pdf_path(document_id)
            if not pdf_path:
                logger.warning(f"Could not find PDF for document {document_id}")
                return None
            
            # Open the PDF document
            doc = fitz.open(pdf_path)
            
            # Make sure the page index is valid
            if page_idx < 0 or page_idx >= len(doc):
                logger.warning(f"Invalid page index {page_idx} for document {document_id}")
                return None
            
            # Get the specified page
            page = doc[page_idx]
            
            # Add padding to the bbox
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - padding / zoom)
            y1 = max(0, y1 - padding / zoom)
            x2 = min(page.rect.width, x2 + padding / zoom)
            y2 = min(page.rect.height, y2 + padding / zoom)
            
            # Create a rectangle for highlighting
            highlight_rect = fitz.Rect(x1, y1, x2, y2)
            
            # Add the highlight to the page (without saving the PDF)
            annot = page.add_highlight_annot(highlight_rect)
            annot.set_colors(stroke=highlight_color)
            annot.update()
            
            # Create a pixmap of just the region we're interested in
            # We adjust the matrix for higher resolution
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=highlight_rect)
            
            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert image to base64 for HTML display
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Clean up
            doc.close()
            
            return img_str
            
        except Exception as e:
            logger.error(f"Error generating thumbnail: {str(e)}")
            return None
    
    def _find_pdf_path(self, document_id):
        """
        Find the PDF file for a document, focusing on the document's directory.
        
        Args:
            document_id: Document ID
            
        Returns:
            Path to the PDF file or None if not found
        """
        # 1. Check document's own directory first (this is where we should always find it now)
        doc_dir = os.path.join(self.storage_dir, document_id)
        if os.path.exists(doc_dir):
            # First look for the standard filename we now use during upload
            original_pdf = os.path.join(doc_dir, "original.pdf")
            if os.path.exists(original_pdf):
                return original_pdf
                
            # Then look for any PDF in the document directory as a fallback
            for filename in os.listdir(doc_dir):
                if filename.lower().endswith(".pdf"):
                    return os.path.join(doc_dir, filename)
        
        # 2. If not found in document directory, check original file path in metadata
        metadata_path = os.path.join(doc_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                # Check original file path
                if "original_file_path" in metadata:
                    orig_path = metadata.get("original_file_path")
                    if os.path.exists(orig_path) and orig_path.lower().endswith(".pdf"):
                        return orig_path
            except Exception as e:
                logger.error(f"Error reading metadata: {str(e)}")
        
        logger.warning(f"Could not find PDF file for document {document_id}")
        return None
    
    def _get_font(self, size=16):
        """
        Safely get a font for PIL text drawing.
        
        Args:
            size: Font size
            
        Returns:
            PIL ImageFont object
        """
        from PIL import ImageFont
        
        # List of font names to try in order of preference
        font_names = [
            "Arial", "DejaVuSans", "FreeSans", "Liberation Sans", 
            "Verdana", "Helvetica", "Tahoma", "Roboto"
        ]
        
        for font_name in font_names:
            try:
                return ImageFont.truetype(font_name, size)
            except IOError:
                continue
        
        # If no system fonts work, use default
        try:
            return ImageFont.load_default()
        except Exception:
            # Last resort - PIL's default font API has changed in different versions
            return None
    
    def _draw_text_with_background(self, draw, text, position, font=None, 
                                  text_color=(255, 255, 255), 
                                  bg_color=(0, 0, 0, 180)):
        """
        Draw text with a background on an image.
        
        Args:
            draw: PIL ImageDraw object
            text: Text to draw
            position: (x, y) position
            font: PIL font to use
            text_color: Text color as RGB tuple
            bg_color: Background color as RGBA tuple
        """
        x, y = position
        
        # Measure text size
        # PIL's text size API has changed in different versions
        if hasattr(draw, 'textsize') and font:
            text_width, text_height = draw.textsize(text, font=font)
        elif hasattr(draw, 'textbbox') and font:
            _, _, text_width, text_height = draw.textbbox((0, 0), text, font=font)
        else:
            # Estimate if we can't measure
            text_width, text_height = len(text) * 8, 16
        
        # Draw background rectangle
        draw.rectangle([x, y, x + text_width + 10, y + text_height + 10], fill=bg_color)
        
        # Draw text
        draw.text((x + 5, y + 5), text, fill=text_color, font=font)
        
    def get_multi_highlight_thumbnail(self, document_id, highlights, 
                                     highlight_color=None,
                                     zoom=1.5, padding=20,
                                     max_width=800, max_height=1200):
        """
        Generate a thumbnail with multiple highlights across multiple pages.
        Stacks page thumbnails vertically into a single image.
        
        Args:
            document_id: Document ID
            highlights: List of {original_page_index, bbox} dicts
            highlight_color: RGBA tuple for highlight color
            zoom: Zoom factor
            padding: Padding around highlights
            max_width/max_height: Maximum dimensions of output thumbnail
            
        Returns:
            Base64 encoded image string or None if error
        """
        # Group highlights by page
        pages = {}
        for hl in highlights:
            page_idx = hl.get("original_page_index", 0)
            if page_idx not in pages:
                pages[page_idx] = []
            pages[page_idx].append(hl.get("bbox"))
        
        # Handle potential empty or invalid bboxes
        valid_pages = {}
        for page_idx, bboxes in pages.items():
            # Filter out None or empty bboxes
            valid_bboxes = [bbox for bbox in bboxes if bbox and len(bbox) == 4]
            if valid_bboxes:
                valid_pages[page_idx] = valid_bboxes
        pages = valid_pages
        
        # Add a maximum number of pages to render to avoid performance issues
        max_pages_to_render = 5  # Limit number of pages to avoid huge images
        if len(pages) > max_pages_to_render:
            logger.warning(f"Limiting thumbnail to {max_pages_to_render} pages out of {len(pages)}")
            # Keep only the first max_pages_to_render pages
            sorted_pages = sorted(pages.keys())[:max_pages_to_render]
            pages = {idx: pages[idx] for idx in sorted_pages}
        
        # If no highlights found, return None
        if not pages:
            return None
        
        try:
            # Find PDF path
            pdf_path = self._find_pdf_path(document_id)
            if not pdf_path:
                return None
                
            # Open document
            doc = fitz.open(pdf_path)
            
            # Generate thumbnail for each page
            page_images = []
            
            # Sort pages by index for consistent order
            sorted_page_indices = sorted(pages.keys())
            
            for page_idx in sorted_page_indices:
                # Skip invalid page indices
                if page_idx < 0 or page_idx >= len(doc):
                    logger.warning(f"Invalid page index {page_idx} for document {document_id}")
                    continue
                    
                bboxes = pages[page_idx]
                if not bboxes:
                    continue
                    
                page = doc[page_idx]
                
                # Calculate the union of all bboxes for this page
                x1, y1, x2, y2 = bboxes[0]
                
                # Expand to include all highlights on this page
                for bbox in bboxes[1:]:
                    x1 = min(x1, bbox[0])
                    y1 = min(y1, bbox[1])
                    x2 = max(x2, bbox[2])
                    y2 = max(y2, bbox[3])
                
                # Add padding
                x1 = max(0, x1 - padding / zoom)
                y1 = max(0, y1 - padding / zoom)
                x2 = min(page.rect.width, x2 + padding / zoom)
                y2 = min(page.rect.height, y2 + padding / zoom)
                
                # Create a clip rectangle for the unified area
                clip_rect = fitz.Rect(x1, y1, x2, y2)
                
                # Add all highlights to the page
                if highlight_color is None:
                    highlight_color = self.highlight_color
                for bbox in bboxes:
                    highlight_rect = fitz.Rect(bbox)
                    annot = page.add_highlight_annot(highlight_rect)
                    annot.set_colors(stroke=highlight_color)
                    annot.update()
                
                # Render the region
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Add page number overlay
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                
                # Get font
                font = self._get_font(size=16)
                
                # Draw page number with background
                label = f"Page {page_idx + 1}"
                self._draw_text_with_background(draw, label, (5, 5), font=font)
                
                # Add the image to our list
                page_images.append(img)
            
            # Clean up the document
            doc.close()
            
            if not page_images:
                logger.warning(f"No valid page images generated for document {document_id}")
                return None
            
            # Combine the page images into a single vertical stack
            total_width = max(img.width for img in page_images)
            total_height = sum(img.height for img in page_images) + (10 * (len(page_images) - 1))  # 10px spacing
            
            # Create a new image with the combined dimensions
            combined_img = Image.new('RGB', (total_width, total_height), color='white')
            
            # Paste each page image
            y_offset = 0
            for img in page_images:
                combined_img.paste(img, (0, y_offset))
                y_offset += img.height + 10  # 10px spacing between pages
            
            # Resize if larger than max dimensions while maintaining aspect ratio
            if combined_img.width > max_width or combined_img.height > max_height:
                combined_img.thumbnail((max_width, max_height), Image.LANCZOS)
            
            # Convert to base64
            buffered = BytesIO()
            combined_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str
                
        except Exception as e:
            logger.error(f"Error generating multi-page thumbnail: {str(e)}", exc_info=True)
            return None
    
    def copy_pdf_to_document_directory(self, document_id):
        """
        Find and copy the original PDF to the document's directory.
        This should be called when loading a document to ensure
        the PDF is available for highlighting.
        
        Args:
            document_id: Document ID
            
        Returns:
            Path to the PDF in document directory or None if copy failed
        """
        doc_dir = os.path.join(self.storage_dir, document_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        # Check if PDF already exists in document directory
        for filename in os.listdir(doc_dir):
            if filename.lower().endswith(".pdf"):
                return os.path.join(doc_dir, filename)
        
        # Try to find PDF from metadata
        metadata_path = os.path.join(doc_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                # Get original file path from metadata
                if "original_file_path" in metadata:
                    orig_path = metadata.get("original_file_path")
                    if os.path.exists(orig_path) and orig_path.lower().endswith(".pdf"):
                        # Copy to document directory
                        target_path = os.path.join(doc_dir, "original.pdf")
                        try:
                            import shutil
                            shutil.copy2(orig_path, target_path)
                            logger.info(f"Copied original PDF from {orig_path} to {target_path}")
                            return target_path
                        except Exception as e:
                            logger.error(f"Error copying PDF from {orig_path}: {str(e)}")
            except Exception as e:
                logger.error(f"Error reading metadata: {str(e)}")
        
        # Look for PDFs with matching names in project directory
        base_name = document_id.split('_')[-1] if '_' in document_id else document_id
        
        # Check parent directory (project root)
        parent_dir = os.path.dirname(os.path.abspath(self.storage_dir))
        for filename in os.listdir(parent_dir):
            if filename.lower().endswith(".pdf") and base_name.lower() in filename.lower():
                orig_path = os.path.join(parent_dir, filename)
                target_path = os.path.join(doc_dir, "original.pdf")
                try:
                    import shutil
                    shutil.copy2(orig_path, target_path)
                    logger.info(f"Copied PDF from {orig_path} to {target_path}")
                    return target_path
                except Exception as e:
                    logger.error(f"Error copying PDF from {orig_path}: {str(e)}")
        
        # If still not found, check common locations
        for pdf_dir_name in ["pdfs", "pdf", "PDF", "documents"]:
            pdfs_dir = os.path.join(parent_dir, pdf_dir_name)
            if os.path.exists(pdfs_dir) and os.path.isdir(pdfs_dir):
                for filename in os.listdir(pdfs_dir):
                    if filename.lower().endswith(".pdf") and base_name.lower() in filename.lower():
                        orig_path = os.path.join(pdfs_dir, filename)
                        target_path = os.path.join(doc_dir, "original.pdf")
                        try:
                            import shutil
                            shutil.copy2(orig_path, target_path)
                            logger.info(f"Copied PDF from {orig_path} to {target_path}")
                            return target_path
                        except Exception as e:
                            logger.error(f"Error copying PDF from {orig_path}: {str(e)}")
        
        logger.warning(f"Could not find any PDF to copy for document {document_id}")
        return None
    
    def debug_pdf_info(self, document_id):
        """
        Get comprehensive debug information about a PDF document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {
            "document_id": document_id,
            "pdf_path": None,
            "pdf_exists": False,
            "page_count": 0,
            "metadata_path": None,
            "metadata_exists": False,
            "metadata": None,
            "doc_dir_exists": False,
            "search_locations": []
        }
        
        # Extract potential base filenames
        doc_parts = document_id.split('_', 2)
        if len(doc_parts) >= 3:
            base_name = doc_parts[2]  # For format like "doc_12345_filename.pdf"
        elif len(doc_parts) == 2:
            base_name = doc_parts[1]  # For format like "doc_filename.pdf"
        else:
            base_name = document_id
        
        debug_info["base_name"] = base_name
        
        # 1. Check document directory
        doc_dir = os.path.join(self.storage_dir, document_id)
        debug_info["doc_dir"] = doc_dir
        debug_info["doc_dir_exists"] = os.path.exists(doc_dir)
        debug_info["search_locations"].append({"location": doc_dir, "exists": os.path.exists(doc_dir)})
        
        # 2. Check metadata
        metadata_path = os.path.join(doc_dir, "metadata.json")
        debug_info["metadata_path"] = metadata_path
        debug_info["metadata_exists"] = os.path.exists(metadata_path)
        
        original_filename = None
        if debug_info["metadata_exists"]:
            try:
                import json
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                debug_info["metadata"] = metadata
                
                # Extract original filename if available
                if "original_filename" in metadata:
                    original_filename = metadata["original_filename"]
                    debug_info["original_filename"] = original_filename
                    
                # Check if there's a direct file path in metadata
                if "original_file_path" in metadata:
                    original_path = metadata["original_file_path"]
                    debug_info["original_file_path"] = original_path
                    debug_info["original_path_exists"] = os.path.exists(original_path)
                    debug_info["search_locations"].append({"location": original_path, "exists": os.path.exists(original_path)})
            except Exception as e:
                debug_info["metadata_error"] = str(e)
        
        # 3. Look for potential PDF locations
        potential_locations = []
        
        # 3.1. Document directory
        if debug_info["doc_dir_exists"]:
            for filename in os.listdir(doc_dir):
                if filename.lower().endswith(".pdf"):
                    potential_locations.append(os.path.join(doc_dir, filename))
        
        # 3.2. PDF directories
        for pdf_dir_name in ["pdfs", "pdf", "PDF", "documents"]:
            pdfs_dir = os.path.join(self.storage_dir, pdf_dir_name)
            if os.path.exists(pdfs_dir):
                debug_info["search_locations"].append({"location": pdfs_dir, "exists": True})
                
                # Try different filename patterns
                potential_files = [
                    f"{document_id}.pdf",
                    f"{base_name}.pdf"
                ]
                
                if original_filename and original_filename.lower().endswith(".pdf"):
                    potential_files.append(original_filename)
                    
                for potential_file in potential_files:
                    potential_path = os.path.join(pdfs_dir, potential_file)
                    debug_info["search_locations"].append({"location": potential_path, "exists": os.path.exists(potential_path)})
                    if os.path.exists(potential_path):
                        potential_locations.append(potential_path)
                        
                # Look for any files containing the base name
                for filename in os.listdir(pdfs_dir):
                    if base_name.lower() in filename.lower() and filename.lower().endswith(".pdf"):
                        potential_path = os.path.join(pdfs_dir, filename)
                        if potential_path not in potential_locations:
                            potential_locations.append(potential_path)
        
        # 3.3. Parent directory
        parent_dir = os.path.dirname(os.path.abspath(self.storage_dir))
        debug_info["parent_dir"] = parent_dir
        debug_info["search_locations"].append({"location": parent_dir, "exists": os.path.exists(parent_dir)})
        
        for filename in os.listdir(parent_dir):
            if filename.lower().endswith(".pdf"):
                if document_id.lower() in filename.lower() or base_name.lower() in filename.lower():
                    potential_path = os.path.join(parent_dir, filename)
                    potential_locations.append(potential_path)
        
        # Store all potential locations
        debug_info["potential_pdf_locations"] = potential_locations
        
        # Try to find PDF path
        pdf_path = self._find_pdf_path(document_id)
        debug_info["pdf_path"] = pdf_path
        debug_info["pdf_exists"] = pdf_path is not None and os.path.exists(pdf_path)
        
        # Get page count if PDF exists
        if debug_info["pdf_exists"]:
            try:
                doc = fitz.open(pdf_path)
                debug_info["page_count"] = len(doc)
                debug_info["pdf_size"] = os.path.getsize(pdf_path)
                doc.close()
            except Exception as e:
                debug_info["pdf_error"] = str(e)
        
        return debug_info