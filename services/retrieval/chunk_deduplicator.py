import hashlib
import logging
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class ChunkDeduplicator:
    """
    Enhanced chunk deduplication utility that handles multiple search methods
    and ensures exactly top_k unique chunks are returned.
    """
    
    def __init__(self, similarity_threshold: float = 0.95):
        """
        Args:
            similarity_threshold: Threshold for considering chunks as similar
        """
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)

    
    def deduplicate_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        top_k: int,
        score_merge_strategy: str = "max",
        enable_similarity_check: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Enhanced deduplication with optional similarity checking.
        
        Args:
            chunks: List of chunk dictionaries
            top_k: Number of unique chunks to return
            score_merge_strategy: How to merge scores
            enable_similarity_check: Whether to check content similarity for near-duplicates
        """
        if not chunks:
            return []
        
        # Step 1: Remove exact duplicates
        exact_groups = self._group_chunks_by_identity(chunks)
        
        # Step 2: Merge exact duplicates
        merged_chunks = []
        for chunk_identity, chunk_list in exact_groups.items():
            merged_chunk = self._merge_duplicate_chunks(chunk_list, score_merge_strategy)
            merged_chunks.append(merged_chunk)
        
        # Step 3: Optional similarity-based deduplication for near-duplicates
        if enable_similarity_check and len(merged_chunks) > top_k:
            merged_chunks = self._remove_similar_chunks(merged_chunks, top_k)
        
        # Step 4: Sort by score and take top_k
        sorted_chunks = sorted(
            merged_chunks, 
            key=lambda x: x.get("score", 0), 
            reverse=True
        )[:top_k]
        
        # Step 5: Add ranking information
        for i, chunk in enumerate(sorted_chunks):
            chunk["retrieval_rank"] = i + 1
            chunk["final_score"] = chunk.get("score", 0)
        
        self._log_deduplication_stats(len(chunks), len(merged_chunks), len(sorted_chunks), top_k)
        
        return sorted_chunks
    
    def _group_chunks_by_identity(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group chunks by their unique identity (content + document + position).
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary mapping chunk identity to list of duplicate chunks
        """
        chunk_groups = defaultdict(list)
        
        for chunk in chunks:
            identity = self._get_chunk_identity(chunk)
            chunk_groups[identity].append(chunk)
        
        return dict(chunk_groups)
    
    def _get_chunk_identity(self, chunk: Dict[str, Any]) -> str:
        """
        Generate a unique identity for a chunk based on content and metadata.
        
        Args:
            chunk: Chunk dictionary
            
        Returns:
            Unique identity string
        """
        # Use multiple factors to create unique identity
        content = chunk.get("content", "")
        document_id = chunk.get("document_id", "")
        
        # Get positional information if available
        metadata = chunk.get("metadata", {})
        page_idx = metadata.get("page_idx", "")
        original_index = metadata.get("original_index", "")
        
        # Get tree level for hierarchical chunks
        tree_level = chunk.get("tree_level", 0)
        
        # Create a more robust content hash (full content, not just first 100 chars)
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
        
        # Combine all identifying factors
        identity_parts = [
            str(document_id),
            str(page_idx),
            str(original_index),
            str(tree_level),
            content_hash
        ]
        
        return "|".join(identity_parts)
    
    def _merge_duplicate_chunks(
        self, 
        chunk_list: List[Dict[str, Any]], 
        score_merge_strategy: str
    ) -> Dict[str, Any]:
        """
        Merge duplicate chunks into a single chunk with combined metadata.
        
        Args:
            chunk_list: List of duplicate chunks
            score_merge_strategy: How to merge scores
            
        Returns:
            Single merged chunk
        """
        if len(chunk_list) == 1:
            return chunk_list[0]
        
        # Use the first chunk as base
        base_chunk = chunk_list[0].copy()
        
        # Collect scores and search methods
        scores = [chunk.get("score", 0) for chunk in chunk_list]
        search_methods = [chunk.get("search_method", "unknown") for chunk in chunk_list]
        
        # Merge scores based on strategy
        if score_merge_strategy == "max":
            merged_score = max(scores)
        elif score_merge_strategy == "avg":
            merged_score = sum(scores) / len(scores)
        elif score_merge_strategy == "sum":
            merged_score = sum(scores)
        else:
            merged_score = max(scores)  # Default to max
        
        # Update the base chunk with merged information
        base_chunk["score"] = merged_score
        base_chunk["search_method"] = "hybrid" if len(set(search_methods)) > 1 else search_methods[0]
        
        # Add detailed search method information
        base_chunk["search_methods_detail"] = {
            "methods": list(set(search_methods)),
            "individual_scores": dict(zip(search_methods, scores)),
            "merge_strategy": score_merge_strategy,
            "duplicate_count": len(chunk_list)
        }
        
        # Merge any additional scores (vector_score, bm25_score, etc.)
        if any("vector_score" in chunk for chunk in chunk_list):
            vector_scores = [chunk.get("vector_score") for chunk in chunk_list if chunk.get("vector_score") is not None]
            if vector_scores:
                base_chunk["vector_score"] = max(vector_scores) if score_merge_strategy == "max" else sum(vector_scores) / len(vector_scores)
        
        if any("bm25_score" in chunk for chunk in chunk_list):
            bm25_scores = [chunk.get("bm25_score") for chunk in chunk_list if chunk.get("bm25_score") is not None]
            if bm25_scores:
                base_chunk["bm25_score"] = max(bm25_scores) if score_merge_strategy == "max" else sum(bm25_scores) / len(bm25_scores)
        
        return base_chunk
    def _remove_similar_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        target_count: int
    ) -> List[Dict[str, Any]]:
        """
        Remove chunks with very similar content to avoid redundancy.
        
        Args:
            chunks: List of chunks to check for similarity
            target_count: Target number of chunks to return
            
        Returns:
            List with similar chunks removed
        """
        if len(chunks) <= target_count:
            return chunks
        
        # Sort by score first
        sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
        
        filtered_chunks = []
        seen_content_hashes = set()
        
        for chunk in sorted_chunks:
            content = chunk.get("content", "")
            
            # Create multiple hash variants to catch similar content
            content_variants = [
                content,
                content.lower(),
                ' '.join(content.split()),  # Normalize whitespace
                content[:200],  # First 200 chars
                content[-200:]  # Last 200 chars
            ]
            
            # Check if any variant is too similar to existing chunks
            is_similar = False
            for variant in content_variants:
                variant_hash = hashlib.md5(variant.encode('utf-8')).hexdigest()
                if variant_hash in seen_content_hashes:
                    is_similar = True
                    break
            
            if not is_similar:
                # Add all variants to seen hashes
                for variant in content_variants:
                    variant_hash = hashlib.md5(variant.encode('utf-8')).hexdigest()
                    seen_content_hashes.add(variant_hash)
                
                filtered_chunks.append(chunk)
                
                if len(filtered_chunks) >= target_count:
                    break
        
        return filtered_chunks
    
    def _log_deduplication_stats(
        self, 
        original_count: int, 
        merged_count: int, 
        final_count: int, 
        requested_top_k: int
    ):
        """Log detailed deduplication statistics."""
        exact_duplicates_removed = original_count - merged_count
        similarity_duplicates_removed = merged_count - final_count
        
        self.logger.info(
            f"Deduplication results: "
            f"{original_count} original → "
            f"{merged_count} after exact dedup (-{exact_duplicates_removed}) → "
            f"{final_count} final (-{similarity_duplicates_removed}) "
            f"[requested: {requested_top_k}]"
        )
        
        if exact_duplicates_removed > 0:
            self.logger.info(f"Removed {exact_duplicates_removed} exact duplicate chunks")
        
        if similarity_duplicates_removed > 0:
            self.logger.info(f"Removed {similarity_duplicates_removed} similar chunks")

# Additional utility for debugging chunk retrieval
class ChunkRetrievalDebugger:
    """Utility for debugging chunk retrieval and deduplication issues."""
    
    @staticmethod
    def analyze_chunk_overlap(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze overlap between chunks to understand duplication patterns.
        
        Returns:
            Analysis report with overlap statistics
        """
        if not chunks:
            return {"total_chunks": 0, "analysis": "No chunks to analyze"}
        
        # Group by document
        by_document = defaultdict(list)
        for chunk in chunks:
            doc_id = chunk.get("document_id", "unknown")
            by_document[doc_id].append(chunk)
        
        # Analyze search methods
        search_methods = defaultdict(int)
        for chunk in chunks:
            method = chunk.get("search_method", "unknown")
            search_methods[method] += 1
        
        # Find potential duplicates by content similarity
        content_lengths = [len(chunk.get("content", "")) for chunk in chunks]
        
        analysis = {
            "total_chunks": len(chunks),
            "documents_covered": len(by_document),
            "chunks_per_document": {doc_id: len(chunks) for doc_id, chunks in by_document.items()},
            "search_methods": dict(search_methods),
            "content_length_stats": {
                "min": min(content_lengths) if content_lengths else 0,
                "max": max(content_lengths) if content_lengths else 0,
                "avg": sum(content_lengths) / len(content_lengths) if content_lengths else 0
            }
        }
        
        return analysis
    
    @staticmethod
    def find_duplicate_patterns(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find patterns in duplicate chunks to help debug deduplication issues.
        """
        # Group by content hash
        content_groups = defaultdict(list)
        for chunk in chunks:
            content = chunk.get("content", "")
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
            content_groups[content_hash].append(chunk)
        
        # Find duplicates
        duplicates = []
        for content_hash, chunk_list in content_groups.items():
            if len(chunk_list) > 1:
                duplicates.append({
                    "content_hash": content_hash,
                    "duplicate_count": len(chunk_list),
                    "chunks": chunk_list,
                    "search_methods": list(set(chunk.get("search_method", "unknown") for chunk in chunk_list)),
                    "documents": list(set(chunk.get("document_id", "unknown") for chunk in chunk_list))
                })
        
        return duplicates