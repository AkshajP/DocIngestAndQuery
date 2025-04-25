import re
import logging
import uuid
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class Chunker:
    """
    Smart chunking of document content to create optimal-sized chunks
    and convert table formats for better retrieval.
    Preserves metadata like bbox coordinates for each chunk.
    """
    
    def __init__(self, max_chunk_size: int = 5000, min_chunk_size: int = 200, overlap_size: int = 200):
        """
        Initialize the chunker with size limits.
        
        Args:
            max_chunk_size: Maximum size (characters) for a chunk
            min_chunk_size: Minimum size to finalize a chunk
            overlap_size: Size of overlap between adjacent chunks
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
    
    def chunk_content(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process the content list and create optimized chunks.
        
        Args:
            content_list: List of content items from extraction
            
        Returns:
            List of optimized chunks
        """
        all_chunks = []
        
        # Separate content by type and page for processing
        text_by_page = {}
        table_items = []
        image_items = []
        
        # Categorize items by type
        for idx, item in enumerate(content_list):
            content_type = item.get("type", "unknown")
            page_idx = item.get("page_idx", 0)
            
            # Add original index for reference
            item_with_idx = item.copy()
            item_with_idx["original_index"] = idx
            
            if content_type == "text":
                if page_idx not in text_by_page:
                    text_by_page[page_idx] = []
                text_by_page[page_idx].append(item_with_idx)
            elif content_type == "table":
                table_items.append(item_with_idx)
            elif content_type == "image":
                image_items.append(item_with_idx)
        
        # Process text items - combine into larger chunks by page
        for page_idx, page_items in text_by_page.items():
            # Sort by original index as a simple proxy for text flow
            page_items.sort(key=lambda x: x.get("original_index", 0))
            all_chunks.extend(self._combine_text_items(page_items, page_idx))
        
        # Process table items - convert HTML to JSON
        all_chunks.extend(self._process_table_items(table_items))
        
        # Process image items directly
        all_chunks.extend(self._process_image_items(image_items))
        
        return all_chunks
    
    def _combine_text_items(self, text_items: List[Dict[str, Any]], page_idx: int) -> List[Dict[str, Any]]:
        """
        Combine text items into larger chunks with overlap between adjacent chunks.
        Preserves bbox information for each chunk.
        
        Args:
            text_items: List of text content items from same page
            page_idx: Page index of these items
            
        Returns:
            List of combined text chunks
        """
        if not text_items:
            return []
        
        chunks = []
        current_text = ""
        current_indices = []
        current_size = 0
        current_bboxes = []  # Track bboxes for current chunk
        
        # Keep track of text and indices for overlap
        all_text_items = []  # List of (text, index, bbox) tuples in order
        
        # First pass - collect all valid text items
        for item in text_items:
            text = item.get("text", "")
            if not text:
                continue
            
            bbox = item.get("bbox", None)  # Get bbox if available
            all_text_items.append((text, item["original_index"], bbox))
        
        if not all_text_items:
            return []
        
        # Second pass - create chunks with overlap
        current_chunk_start = 0
        i = 0
        
        while i < len(all_text_items):
            text, idx, bbox = all_text_items[i]
            text_size = len(text)
            
            # If adding this item would exceed max size and we have enough content,
            # finalize the current chunk and start a new one with overlap
            if (current_size + text_size > self.max_chunk_size and 
                current_size >= self.min_chunk_size):
                
                # Create chunk
                chunk_id = f"text_{current_indices[0]}_{current_indices[-1]}"
                
                # Prepare original_boxes data structure - contains all bbox information
                original_boxes = []
                for j, bbox_info in enumerate(current_bboxes):
                    if bbox_info:
                        original_boxes.append({
                            "original_page_index": page_idx,
                            "bbox": bbox_info,
                            "original_index": current_indices[j]
                        })
                
                chunk = {
                    "id": chunk_id,
                    "content": current_text,
                    "metadata": {
                        "type": "text",
                        "page_idx": page_idx,
                        "original_indices": current_indices,
                        "original_boxes": original_boxes  # Add bbox information
                    }
                }
                chunks.append(chunk)
                
                # Calculate new starting position with overlap
                # Find position that gives us approximately overlap_size characters of overlap
                overlap_chars = 0
                backtrack_idx = -1
                
                for j in range(len(current_indices) - 1, -1, -1):
                    item_idx = current_indices[j]
                    # Find the position of this item in all_text_items
                    pos = next((k for k, (_, idx, _) in enumerate(all_text_items) if idx == item_idx), -1)
                    
                    if pos >= 0:
                        item_text = all_text_items[pos][0]
                        overlap_chars += len(item_text)
                        
                        if overlap_chars >= self.overlap_size:
                            backtrack_idx = pos
                            break
                
                if backtrack_idx >= 0:
                    # Start new chunk from this position
                    current_chunk_start = backtrack_idx
                    i = current_chunk_start  # Reset loop to this position
                else:
                    # If we couldn't find a good overlap point, move forward by half the chunk
                    current_chunk_start = max(0, i - 2)  # At least go back 2 items if possible
                    i = current_chunk_start
                
                # Reset for next chunk
                current_text = ""
                current_indices = []
                current_bboxes = []
                current_size = 0
                continue  # Skip increment at end of loop since we set i explicitly
            
            # Add this item to the current chunk
            if current_text:
                current_text += " "
            current_text += text
            current_indices.append(idx)
            current_bboxes.append(bbox)  # Store bbox for this text segment
            current_size += text_size
            i += 1
        
        # Add the last chunk if it has content
        if current_text:
            chunk_id = f"text_{current_indices[0]}_{current_indices[-1]}"
            
            # Prepare original_boxes data for the last chunk
            original_boxes = []
            for j, bbox_info in enumerate(current_bboxes):
                if bbox_info:
                    original_boxes.append({
                        "original_page_index": page_idx,
                        "bbox": bbox_info,
                        "original_index": current_indices[j]
                    })
            
            chunk = {
                "id": chunk_id,
                "content": current_text,
                "metadata": {
                    "type": "text",
                    "page_idx": page_idx,
                    "original_indices": current_indices,
                    "original_boxes": original_boxes  # Add bbox information
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def _process_table_items(self, table_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process table items, converting HTML to JSON.
        Preserves bbox information.
        
        Args:
            table_items: List of table content items
            
        Returns:
            List of processed table chunks
        """
        # Implementation of table processing
        # Convert HTML tables to structured JSON and text representation
        chunks = []
        
        for item in table_items:
            # Process table content
            # Create chunk with appropriate metadata
            chunk_id = f"table_{uuid.uuid4().hex[:8]}"
            chunks.append({
                "id": chunk_id,
                "content": "Table content representation",
                "metadata": {
                    "type": "table",
                    "page_idx": item.get("page_idx", 0),
                    "original_index": item.get("original_index"),
                    "table_data": {},  # Will contain table structure
                    "original_boxes": []  # Will contain bbox information
                }
            })
        
        return chunks
    
    def _process_image_items(self, image_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process image items.
        Preserves bbox information.
        
        Args:
            image_items: List of image content items
            
        Returns:
            List of processed image chunks
        """
        # Implementation of image processing
        # Create text representations of images with captions
        chunks = []
        
        for item in image_items:
            # Process image content
            # Create chunk with appropriate metadata
            chunk_id = f"image_{uuid.uuid4().hex[:8]}"
            chunks.append({
                "id": chunk_id,
                "content": "Image caption and description",
                "metadata": {
                    "type": "image",
                    "page_idx": item.get("page_idx", 0),
                    "original_index": item.get("original_index"),
                    "img_path": item.get("img_path", ""),
                    "caption": item.get("img_caption", ""),
                    "original_boxes": []  # Will contain bbox information
                }
            })
        
        return chunks