import re
import logging
import uuid
from typing import List, Dict, Any, Optional
import os
logger = logging.getLogger(__name__)
from datetime import datetime

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
        FIXED VERSION: Prevents infinite loops and adds progress tracking.
        """
        if not text_items:
            return []
        
        chunks = []
        current_text = ""
        current_indices = []
        current_size = 0
        current_bboxes = []
        
        # Collect all valid text items first
        all_text_items = []
        for item in text_items:
            text = item.get("text", "")
            if not text:
                continue
            
            bbox = item.get("bbox", None)
            all_text_items.append((text, item["original_index"], bbox))
        
        if not all_text_items:
            return []
        
        logger.info(f"Processing {len(all_text_items)} text items for page {page_idx}")
        
        # SAFETY MEASURES: Prevent infinite loops
        i = 0
        safety_counter = 0
        MAX_ITERATIONS = len(all_text_items) * 3  # Safety limit
        last_i = -1  # Track if i is advancing
        
        while i < len(all_text_items) and safety_counter < MAX_ITERATIONS:
            safety_counter += 1
            
            # SAFETY CHECK: Ensure we're making progress
            if i == last_i:
                logger.warning(f"Chunking loop detected same index {i} twice, forcing advance")
                i += 1
                continue
            last_i = i
            
            # Progress logging for large documents
            if safety_counter % 50 == 0:
                logger.info(f"Chunking progress: {safety_counter}/{len(all_text_items)} items processed")
            
            text, idx, bbox = all_text_items[i]
            text_size = len(text)
            
            # If adding this item would exceed max size and we have enough content,
            # finalize the current chunk and start a new one with overlap
            if (current_size + text_size > self.max_chunk_size and 
                current_size >= self.min_chunk_size):
                
                # Create chunk with current content
                if current_text.strip():
                    chunk_id = f"text_{current_indices[0]}_{current_indices[-1]}"
                    
                    # Prepare original_boxes data structure
                    original_boxes = []
                    for j, bbox_info in enumerate(current_bboxes):
                        if bbox_info and isinstance(bbox_info, (list, tuple)) and len(bbox_info) >= 4:
                            original_boxes.append({
                                "original_page_index": page_idx,
                                "bbox": bbox_info,
                                "original_index": current_indices[j] if j < len(current_indices) else idx
                            })
                    
                    chunk = {
                        "id": chunk_id,
                        "content": current_text.strip(),
                        "metadata": {
                            "type": "text",
                            "page_idx": page_idx,
                            "original_indices": current_indices,
                            "original_boxes": original_boxes
                        }
                    }
                    chunks.append(chunk)
                    logger.debug(f"Created chunk {len(chunks)} with {len(current_text)} characters")
                
                # SIMPLIFIED OVERLAP LOGIC: Just go back a few items instead of complex calculation
                overlap_items = min(3, len(current_indices))  # Simple: overlap last 3 items
                if overlap_items > 0:
                    # Find the starting position for overlap
                    overlap_start_idx = current_indices[-overlap_items]
                    # Find this index in our all_text_items array
                    new_i = max(0, i - overlap_items)
                    
                    # Ensure we're moving forward
                    if new_i >= i:
                        new_i = max(0, i - 1)
                    
                    i = new_i
                else:
                    # No overlap possible, just continue from current position
                    pass
                
                # Reset for next chunk
                current_text = ""
                current_indices = []
                current_bboxes = []
                current_size = 0
                continue
            
            # Add this item to the current chunk
            if current_text:
                current_text += " "
            current_text += text
            current_indices.append(idx)
            current_bboxes.append(bbox)
            current_size += text_size
            
            # ENSURE FORWARD PROGRESS
            i += 1
        
        # SAFETY CHECK: Report if we hit limits
        if safety_counter >= MAX_ITERATIONS:
            logger.error(f"Chunking safety limit reached for page {page_idx}. Created {len(chunks)} chunks so far.")
        
        # Add the last chunk if it has content
        if current_text.strip():
            chunk_id = f"text_{current_indices[0]}_{current_indices[-1]}" if current_indices else f"text_final_{page_idx}"
            
            # Prepare original_boxes data for the last chunk
            original_boxes = []
            for j, bbox_info in enumerate(current_bboxes):
                if bbox_info and isinstance(bbox_info, (list, tuple)) and len(bbox_info) >= 4:
                    original_boxes.append({
                        "original_page_index": page_idx,
                        "bbox": bbox_info,
                        "original_index": current_indices[j] if j < len(current_indices) else 0
                    })
            
            chunk = {
                "id": chunk_id,
                "content": current_text.strip(),
                "metadata": {
                    "type": "text",
                    "page_idx": page_idx,
                    "original_indices": current_indices,
                    "original_boxes": original_boxes
                }
            }
            chunks.append(chunk)
            logger.debug(f"Created final chunk {len(chunks)} with {len(current_text)} characters")
        
        logger.info(f"Completed chunking for page {page_idx}: {len(chunks)} chunks created from {len(all_text_items)} text items")
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
        chunks = []
        
        for item in table_items:
            table_body = item.get("table_body", "")
            if not table_body:
                continue
                
            # Get table caption
            table_caption = item.get("table_caption", "")
            if isinstance(table_caption, list):
                table_caption = " ".join(table_caption)
            
            # Convert HTML table to JSON
            table_json = self._html_table_to_json(table_body)
            
            # Create readable text representation for embedding
            table_text = self._create_table_text(table_json, table_caption)
            
            # Create chunk ID
            chunk_id = f"table_{item['original_index']}"
            
            # Get bbox if available
            bbox = item.get("bbox")
            page_idx = item.get("page_idx", 0)
            
            # Create original_boxes structure
            original_boxes = []
            if bbox:
                original_boxes.append({
                    "original_page_index": page_idx,
                    "bbox": bbox,
                    "original_index": item["original_index"]
                })
            
            # Create chunk
            chunk = {
                "id": chunk_id,
                "content": table_text,  # Use text representation as content
                "metadata": {
                    "type": "table",
                    "page_idx": page_idx,
                    "original_index": item["original_index"],
                    "caption": table_caption,
                    "table_data": table_json,  # Store JSON data in metadata
                    "original_boxes": original_boxes  # Add bbox information
                }
            }
            chunks.append(chunk)
        
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
        chunks = []
        
        for item in image_items:
            img_path = item.get("img_path", "")
            if not img_path:
                continue
            
            # Get image caption
            img_caption = item.get("img_caption", "")
            if isinstance(img_caption, list):
                img_caption = " ".join(img_caption)
            
            # Create a textual representation for the image
            content = f"Image: {os.path.basename(img_path)}\n"
            if img_caption:
                content += f"Caption: {img_caption}\n"
            
            # Create chunk ID
            chunk_id = f"image_{item['original_index']}"
            
            # Get bbox if available
            bbox = item.get("bbox")
            page_idx = item.get("page_idx", 0)
            
            # Create original_boxes structure
            original_boxes = []
            if bbox:
                original_boxes.append({
                    "original_page_index": page_idx,
                    "bbox": bbox,
                    "original_index": item["original_index"]
                })
            
            # Create chunk
            chunk = {
                "id": chunk_id,
                "content": content,
                "metadata": {
                    "type": "image",
                    "page_idx": page_idx,
                    "original_index": item["original_index"],
                    "img_path": img_path,
                    "caption": img_caption,
                    "original_boxes": original_boxes  # Add bbox information
                }
            }
            chunks.append(chunk)
        
        return chunks
    def _html_table_to_json(self, html_table: str) -> Dict[str, Any]:
        """
        Convert HTML table to JSON structure.
        
        Args:
            html_table: HTML table string
            
        Returns:
            Dictionary with table data
        """
        import re
        table_json = {"headers": [], "rows": []}
        
        # Extract headers (th elements)
        header_matches = re.findall(r"<th.*?>(.*?)</th>", html_table, re.DOTALL|re.IGNORECASE)
        if header_matches:
            table_json["headers"] = [self._clean_html(h) for h in header_matches]
        
        # Extract rows (tr elements)
        row_matches = re.findall(r"<tr.*?>(.*?)</tr>", html_table, re.DOTALL|re.IGNORECASE)
        for row_html in row_matches:
            # Skip if this is a header row (contains th)
            if "<th" in row_html.lower():
                continue
                
            # Extract cells (td elements)
            cell_matches = re.findall(r"<td.*?>(.*?)</td>", row_html, re.DOTALL|re.IGNORECASE)
            if cell_matches:
                row_data = [self._clean_html(c) for c in cell_matches]
                table_json["rows"].append(row_data)
        
        return table_json

    def _clean_html(self, html_text: str) -> str:
        """
        Clean HTML text by removing tags and normalizing whitespace.
        
        Args:
            html_text: HTML text to clean
            
        Returns:
            Cleaned text
        """
        import re
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", html_text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def _create_table_text(self, table_json: Dict[str, Any], caption: str = "") -> str:
        """
        Create a textual representation of a table for embedding.
        
        Args:
            table_json: Table data in JSON format
            caption: Table caption
            
        Returns:
            Textual representation of the table
        """
        lines = []
        
        # Add caption if available
        if caption:
            lines.append(f"Table: {caption}")
            lines.append("")
        
        # Add headers
        headers = table_json.get("headers", [])
        if headers:
            lines.append(" | ".join(headers))
            lines.append("-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)))
        
        # Add rows
        for row in table_json.get("rows", []):
            lines.append(" | ".join(row))
        
        return "\n".join(lines)