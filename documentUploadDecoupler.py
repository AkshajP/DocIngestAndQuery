import os
import json
import time
import uuid
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import tempfile
import shutil
from datetime import datetime
import re

# Import components from your existing codebase
from mineru_ingester import ingest_pdf
from raptorRetriever import Raptor
from langchain_ollama import OllamaEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('document_uploader')

class DocumentUploader:
    """
    Handles uploading, processing, and persistent storage of documents.
    This component decouples document processing from querying.
    """
    
    def __init__(
        self,
        storage_dir: str = "document_store",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2",
        ollama_embed_model: str = "llama3.2",
        language: str = 'en',
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_tree_levels: int = 3
    ):
        """
        Initialize the document uploader.
        
        Args:
            storage_dir: Base directory for storing processed documents
            ollama_base_url: URL for Ollama API
            ollama_model: Model for LLM operations
            ollama_embed_model: Model for embeddings
            language: Language for OCR and processing
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between text chunks
            max_tree_levels: Maximum levels for RAPTOR tree
        """
        self.storage_dir = storage_dir
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.ollama_embed_model = ollama_embed_model
        self.language = language
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tree_levels = max_tree_levels
        
        # Create main storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Create registry file if it doesn't exist
        self.registry_path = os.path.join(storage_dir, "document_registry.json")
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, "w") as f:
                json.dump({"documents": {}, "last_updated": datetime.now().isoformat()}, f, indent=2)
        
        # Initialize embedding model
        self.embeddings = OllamaEmbeddings(
            model=ollama_embed_model,
            base_url=ollama_base_url
        )
        
        logger.info("DocumentUploader initialized successfully")
    
    def upload_document(self, file_path: str, document_id: Optional[str] = None, 
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Upload and process a document, storing all components to disk.
        
        Args:
            file_path: Path to the document file
            document_id: Optional custom ID (generated if not provided)
            metadata: Optional metadata about the document
            
        Returns:
            Dictionary with upload status and document information
        """
        start_time = time.time()
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = f"doc_{uuid.uuid4().hex[:10]}_{os.path.basename(file_path)}"
        
        # Create document directory structure
        doc_dir = os.path.join(self.storage_dir, document_id)
        
        # Check if document already exists
        if os.path.exists(doc_dir):
            logger.warning(f"Document {document_id} already exists. Using a new ID.")
            document_id = f"doc_{uuid.uuid4().hex[:10]}_{os.path.basename(file_path)}"
            doc_dir = os.path.join(self.storage_dir, document_id)
        
        # Create directory structure
        os.makedirs(doc_dir, exist_ok=True)
        os.makedirs(os.path.join(doc_dir, "chunks"), exist_ok=True)
        os.makedirs(os.path.join(doc_dir, "embeddings"), exist_ok=True)
        os.makedirs(os.path.join(doc_dir, "raptor_tree"), exist_ok=True)
        os.makedirs(os.path.join(doc_dir, "mineru_output"), exist_ok=True)
        
        # Process document based on type
        if file_path.lower().endswith('.pdf'):
            logger.info(f"Processing PDF document: {file_path}")
            try:
                # Step 1: Process with MinerU
                mineru_result = self._process_with_mineru(file_path, doc_dir)
                
                # Step 2: Create chunks from MinerU output
                chunks = self._create_chunks_from_mineru(mineru_result, doc_dir)
                
                # Step 3: Generate embeddings for chunks
                chunk_embeddings = self._generate_embeddings(chunks, doc_dir)
                
                # Step 4: Build RAPTOR tree structure
                raptor_tree = self._build_raptor_tree(chunks, chunk_embeddings, doc_dir)
                
                # Record processing time
                processing_time = time.time() - start_time
                
                # Create document metadata
                doc_metadata = {
                    "document_id": document_id,
                    "original_filename": os.path.basename(file_path),
                    "original_file_path": file_path,
                    "file_type": "pdf",
                    "processing_time": processing_time,
                    "processing_date": datetime.now().isoformat(),
                    "language": self.language,
                    "chunks_count": len(chunks),
                    "raptor_levels": list(raptor_tree.keys()),
                    "user_metadata": metadata or {},
                    "content_types": self._count_content_types(chunks),
                    "status": "processed"
                }
                
                # Save document metadata
                metadata_path = os.path.join(doc_dir, "metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(doc_metadata, f, indent=2)
                
                # Update document registry
                self._update_registry(document_id, doc_metadata)
                
                logger.info(f"Document {document_id} processed successfully in {processing_time:.2f} seconds")
                
                return {
                    "status": "success",
                    "document_id": document_id,
                    "processing_time": processing_time,
                    "chunks_count": len(chunks),
                    "raptor_levels": list(raptor_tree.keys())
                }
                
            except Exception as e:
                logger.error(f"Error processing document {file_path}: {str(e)}", exc_info=True)
                
                # Create failure metadata
                failure_metadata = {
                    "document_id": document_id,
                    "original_filename": os.path.basename(file_path),
                    "original_file_path": file_path,
                    "processing_date": datetime.now().isoformat(),
                    "error": str(e),
                    "status": "failed"
                }
                
                # Save failure metadata
                metadata_path = os.path.join(doc_dir, "metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(failure_metadata, f, indent=2)
                
                # Update document registry
                self._update_registry(document_id, failure_metadata)
                
                return {
                    "status": "error",
                    "document_id": document_id,
                    "error": str(e)
                }
        else:
            # Future support for other document types
            logger.error(f"Unsupported file type: {file_path}")
            return {
                "status": "error",
                "document_id": document_id,
                "error": "Unsupported file type. Only PDF is currently supported."
            }
    
    def _process_with_mineru(self, pdf_path: str, doc_dir: str) -> Dict[str, Any]:
        """
        Process a PDF document with MinerU and save results.
        
        Args:
            pdf_path: Path to the PDF file
            doc_dir: Directory to save outputs
            
        Returns:
            MinerU processing results
        """
        logger.info(f"Processing PDF with MinerU: {pdf_path}")
        
        try:
            # Run MinerU extraction
            mineru_output = ingest_pdf(pdf_path, lang=self.language)
            
            # Save content_list to output directory
            mineru_dir = os.path.join(doc_dir, "mineru_output")
            content_list_path = os.path.join(mineru_dir, "content_list.json")
            with open(content_list_path, "w", encoding="utf-8") as f:
                json.dump(mineru_output["content_list"], f, indent=2)
            
            # Save images if present
            if "images" in mineru_output and mineru_output["images"]:
                images_dir = os.path.join(mineru_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
                
                for img_name, img_data in mineru_output["images"].items():
                    img_path = os.path.join(images_dir, img_name)
                    with open(img_path, "wb") as f:
                        f.write(img_data)
            
            logger.info(f"MinerU extraction complete with {len(mineru_output['content_list'])} content items")
            return mineru_output
            
        except Exception as e:
            logger.error(f"Error in MinerU processing: {str(e)}")
            raise
    
    def _create_chunks_from_mineru(self, mineru_output: Dict[str, Any], doc_dir: str) -> List[Dict[str, Any]]:
        """
        Create document chunks from MinerU output using the Chunker class.
        
        Args:
            mineru_output: Output from MinerU processing
            doc_dir: Document directory
            
        Returns:
            List of chunk objects with content and metadata
        """
        logger.info("Creating optimized chunks from MinerU output")
        
        content_list = mineru_output.get("content_list", [])
        chunks_dir = os.path.join(doc_dir, "chunks")
        
        # Create a Chunker instance with settings from the document uploader
        chunker = Chunker(
            max_chunk_size=5000, 
            min_chunk_size=200, 
            overlap_size=self.chunk_overlap  # Use the overlap setting from the main class
        )
        
        # Use the chunker to process the content list
        all_chunks = chunker.chunk_content(content_list)
        
        # Save individual chunks to files
        for chunk in all_chunks:
            chunk_path = os.path.join(chunks_dir, f"{chunk['id']}.json")
            with open(chunk_path, "w", encoding="utf-8") as f:
                json.dump(chunk, f, indent=2)
        
        # Save all chunks as a single file for convenience
        all_chunks_path = os.path.join(doc_dir, "all_chunks.json")
        with open(all_chunks_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2)
        
        logger.info(f"Created {len(all_chunks)} optimized chunks")
        return all_chunks
    
    def _generate_embeddings(self, chunks: List[Dict[str, Any]], doc_dir: str) -> Dict[str, List[float]]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of document chunks
            doc_dir: Document directory
            
        Returns:
            Dictionary mapping chunk IDs to embeddings
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        embeddings_dir = os.path.join(doc_dir, "embeddings")
        chunk_texts = [chunk["content"] for chunk in chunks]
        chunk_ids = [chunk["id"] for chunk in chunks]
        
        # Generate embeddings in batches to avoid memory issues
        batch_size = 10
        all_embeddings = {}
        
        for i in range(0, len(chunk_texts), batch_size):
            batch_end = min(i + batch_size, len(chunk_texts))
            batch_texts = chunk_texts[i:batch_end]
            batch_ids = chunk_ids[i:batch_end]
            
            # Generate embeddings
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            
            # Process each embedding
            for j, emb in enumerate(batch_embeddings):
                chunk_id = batch_ids[j]
                all_embeddings[chunk_id] = emb
                
                # Save individual embedding file
                emb_path = os.path.join(embeddings_dir, f"{chunk_id}.pkl")
                with open(emb_path, "wb") as f:
                    pickle.dump(emb, f)
        
        # Save all embeddings as a single file
        all_embeddings_path = os.path.join(doc_dir, "all_embeddings.pkl")
        with open(all_embeddings_path, "wb") as f:
            pickle.dump(all_embeddings, f)
        
        logger.info(f"Generated embeddings for {len(all_embeddings)} chunks")
        return all_embeddings
    
    def _build_raptor_tree(
        self, 
        chunks: List[Dict[str, Any]], 
        chunk_embeddings: Dict[str, List[float]],
        doc_dir: str
    ) -> Dict[int, Dict[str, Any]]:
        """
        Build RAPTOR tree structure and save to disk.
        This implementation creates a temporary RAPTOR instance to build the tree.
        
        Args:
            chunks: List of document chunks
            chunk_embeddings: Dictionary of chunk embeddings
            doc_dir: Document directory
            
        Returns:
            Dictionary with RAPTOR tree structure
        """
        logger.info("Building RAPTOR tree structure")
        
        raptor_dir = os.path.join(doc_dir, "raptor_tree")
        
        # Convert chunks and embeddings to format needed for RAPTOR
        texts = [chunk["content"] for chunk in chunks]
        embeddings_list = [chunk_embeddings[chunk["id"]] for chunk in chunks]
        
        # Create a temporary RAPTOR instance just for tree building
        # We don't save the instance itself, just extract the tree structure
        temp_raptor = self._create_temp_raptor_instance()
        
        try:
            # Use internal RAPTOR methods to build tree
            # Simulate the tree building process directly without storing in Milvus
            
            # Step 1: Create DataFrame from chunks for layer 0
            df_layer0 = pd.DataFrame()
            df_layer0["text"] = texts
            df_layer0["embd"] = embeddings_list
            
            # Store the layer 0 dataframe
            layer0_path = os.path.join(raptor_dir, "layer0_clusters.pkl")
            df_layer0.to_pickle(layer0_path)
            
            # Initialize tree structure
            raptor_tree = {}
            
            # Step 2: Recursively build higher layers
            current_texts = texts
            current_level = 1
            
            while current_level <= self.max_tree_levels and len(current_texts) > 1:
                logger.info(f"Building RAPTOR layer {current_level}")
                
                # Use RAPTOR's internal methods to cluster and summarize
                try:
                    # Use a temporary RAPTOR instance to handle this layer
                    df_clusters, df_summary = temp_raptor._embed_cluster_summarize_texts(
                        current_texts, 
                        level=current_level
                    )
                    
                    # Save the clusters and summaries for this layer
                    clusters_path = os.path.join(raptor_dir, f"layer{current_level}_clusters.pkl")
                    df_clusters.to_pickle(clusters_path)
                    
                    summary_path = os.path.join(raptor_dir, f"layer{current_level}_summaries.pkl")
                    df_summary.to_pickle(summary_path)
                    
                    # Also save as JSON for easier inspection
                    clusters_json_path = os.path.join(raptor_dir, f"layer{current_level}_clusters.json")
                    summary_json_path = os.path.join(raptor_dir, f"layer{current_level}_summaries.json")
                    
                    
                    # Convert DataFrame to JSON-serializable format
                    clusters_json = df_clusters.to_dict(orient="records")
                    # Convert any NumPy arrays to lists
                    for item in clusters_json:
                        for key, value in item.items():
                            if isinstance(value, np.ndarray):
                                item[key] = value.tolist()  # Convert ndarray to list

                    summary_json = df_summary.to_dict(orient="records")
                    
                    with open(clusters_json_path, "w") as f:
                        json.dump(clusters_json, f, indent=2)
                    
                    with open(summary_json_path, "w") as f:
                        json.dump(summary_json, f, indent=2)
                    
                    # Store in tree structure
                    raptor_tree[current_level] = {
                        "clusters": clusters_json,
                        "summaries": summary_json
                    }
                    
                    # Use summaries as input for next level
                    current_texts = df_summary["summaries"].tolist()
                    
                    # Stop if we reach only one cluster
                    if df_summary["cluster"].nunique() <= 1:
                        break
                    
                    current_level += 1
                    
                except Exception as e:
                    logger.error(f"Error building RAPTOR layer {current_level}: {str(e)}")
                    break
            
            # Create a tree structure JSON with node relationships
            tree_structure = self._build_tree_structure(chunks, raptor_tree)
            
            # Save tree structure
            tree_structure_path = os.path.join(doc_dir, "tree_structure.json")
            with open(tree_structure_path, "w") as f:
                json.dump(tree_structure, f, indent=2)
            
            logger.info(f"Successfully built RAPTOR tree with {len(raptor_tree)} levels")
            return raptor_tree
            
        except Exception as e:
            logger.error(f"Error building RAPTOR tree: {str(e)}", exc_info=True)
            return {}
        finally:
            # Clean up temporary RAPTOR instance
            if hasattr(temp_raptor, 'cleanup'):
                temp_raptor.cleanup()
    
    def _create_temp_raptor_instance(self):
        """Create a temporary RAPTOR instance for tree building"""
        # Use a random collection name to avoid conflicts
        temp_collection = f"temp_raptor_{uuid.uuid4().hex[:8]}"
        
        # Initialize with minimal configuration just for tree building
        temp_raptor = Raptor(
            ollama_base_url=self.ollama_base_url,
            ollama_model=self.ollama_model,
            ollama_embed_model=self.ollama_embed_model,
            collection_name=temp_collection,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            max_tree_levels=self.max_tree_levels
        )
        
        return temp_raptor
    
    def _build_tree_structure(
        self, 
        chunks: List[Dict[str, Any]], 
        raptor_tree: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build a comprehensive tree structure with parent-child relationships.
        
        Args:
            chunks: Original document chunks
            raptor_tree: RAPTOR tree data
            
        Returns:
            Complete tree structure with node relationships
        """
        # Create node dictionary
        nodes = {}
        
        # Add leaf nodes (original chunks)
        for i, chunk in enumerate(chunks):
            node_id = f"leaf_{i}"
            nodes[node_id] = {
                "id": node_id,
                "content": chunk["content"],
                "level": 0,
                "type": "original",
                "metadata": chunk["metadata"],
                "children": []  # Leaf nodes have no children
            }
        
        # Process each layer of the tree
        for level, level_data in raptor_tree.items():
            clusters = level_data["clusters"]
            summaries = level_data["summaries"]
            
            # Process clusters to establish parent-child relationships
            for i, cluster_info in enumerate(summaries):
                # Create node for this summary
                node_id = f"summary_l{level}_c{cluster_info['cluster']}"
                
                # Find children from the previous level
                children = []
                
                if level == 1:
                    # For level 1, children are leaf nodes
                    for j, cluster_data in enumerate(clusters):
                        if cluster_info["cluster"] in cluster_data.get("cluster", []):
                            children.append(f"leaf_{j}")
                else:
                    # For higher levels, children are summaries from the previous level
                    prev_level_summaries = raptor_tree[level-1]["summaries"]
                    for j, prev_summary in enumerate(prev_level_summaries):
                        # This is a simplification - would need actual cluster matching
                        prev_cluster = prev_summary["cluster"]
                        for cluster_data in clusters:
                            if cluster_info["cluster"] in cluster_data.get("cluster", []):
                                children.append(f"summary_l{level-1}_c{prev_cluster}")
                
                # Create node
                nodes[node_id] = {
                    "id": node_id,
                    "content": cluster_info["summaries"],
                    "level": level,
                    "type": "summary",
                    "metadata": {
                        "cluster": cluster_info["cluster"],
                        "level": level
                    },
                    "children": list(set(children))  # Remove duplicates
                }
        
        # Build full tree structure
        tree_structure = {
            "nodes": nodes,
            "levels": list(raptor_tree.keys()) + [0],  # Include leaf level
            "root_nodes": [node_id for node_id, info in nodes.items() if info["level"] == max(raptor_tree.keys())]
        }
        
        return tree_structure
    
    def _count_content_types(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Count the number of chunks by content type.
        
        Args:
            chunks: List of content chunks
            
        Returns:
            Dictionary with counts by type
        """
        type_counts = {}
        for chunk in chunks:
            chunk_type = chunk.get("metadata", {}).get("type", "unknown")
            if chunk_type in type_counts:
                type_counts[chunk_type] += 1
            else:
                type_counts[chunk_type] = 1
                
        return type_counts
    
    def _update_registry(self, document_id: str, metadata: Dict[str, Any]):
        """
        Update the document registry with new document information.
        
        Args:
            document_id: Document ID
            metadata: Document metadata
        """
        # Load current registry
        with open(self.registry_path, "r") as f:
            registry = json.load(f)
        
        # Update registry with new document
        registry["documents"][document_id] = {
            "document_id": document_id,
            "filename": metadata.get("original_filename", ""),
            "processing_date": metadata.get("processing_date", datetime.now().isoformat()),
            "status": metadata.get("status", "unknown"),
            "chunks_count": metadata.get("chunks_count", 0),
            "content_types": metadata.get("content_types", {}),
            "raptor_levels": metadata.get("raptor_levels", []),
            "file_type": metadata.get("file_type", "unknown")
        }
        
        registry["last_updated"] = datetime.now().isoformat()
        
        # Save updated registry
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the registry.
        
        Returns:
            List of document metadata
        """
        # Load registry
        with open(self.registry_path, "r") as f:
            registry = json.load(f)
        
        # Return list of documents
        return list(registry["documents"].values())
    
    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document metadata or None if document not found
        """
        metadata_path = os.path.join(self.storage_dir, document_id, "metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        
        return None
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its processed data.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        doc_dir = os.path.join(self.storage_dir, document_id)
        
        if not os.path.exists(doc_dir):
            logger.warning(f"Document {document_id} not found")
            return False
        
        try:
            # Remove from registry first
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            
            if document_id in registry["documents"]:
                del registry["documents"][document_id]
                registry["last_updated"] = datetime.now().isoformat()
                
                with open(self.registry_path, "w") as f:
                    json.dump(registry, f, indent=2)
            
            # Delete document directory
            shutil.rmtree(doc_dir)
            
            logger.info(f"Document {document_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False


class Chunker:
    """
    Smart chunking of document content to create optimal-sized chunks
    and convert table formats for better retrieval.
    """
    
    def __init__(self, max_chunk_size=5000, min_chunk_size=200, overlap_size=200):
        """
        Initialize the chunker with size limits.
        
        Args:
            max_chunk_size: Maximum size (in characters) for a chunk
            min_chunk_size: Minimum size (in characters) to finalize a chunk
            overlap_size: Size of overlap between adjacent chunks
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
    
    def chunk_content(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process the content list and create optimized chunks.
        
        Args:
            content_list: List of content items from MinerU output
            
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
        
        # Keep track of text and indices for overlap
        all_text_items = []  # List of (text, index) tuples in order
        
        # First pass - collect all valid text items
        for item in text_items:
            text = item.get("text", "")
            if not text:
                continue
            
            all_text_items.append((text, item["original_index"]))
        
        if not all_text_items:
            return []
        
        # Second pass - create chunks with overlap
        current_chunk_start = 0
        i = 0
        
        while i < len(all_text_items):
            text, idx = all_text_items[i]
            text_size = len(text)
            
            # If adding this item would exceed max size and we have enough content,
            # finalize the current chunk and start a new one with overlap
            if (current_size + text_size > self.max_chunk_size and 
                current_size >= self.min_chunk_size):
                
                # Create chunk
                chunk_id = f"text_{current_indices[0]}_{current_indices[-1]}"
                chunk = {
                    "id": chunk_id,
                    "content": current_text,
                    "metadata": {
                        "type": "text",
                        "page_idx": page_idx,
                        "original_indices": current_indices
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
                    pos = next((k for k, (_, idx) in enumerate(all_text_items) if idx == item_idx), -1)
                    
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
                current_size = 0
                continue  # Skip increment at end of loop since we set i explicitly
            
            # Add this item to the current chunk
            if current_text:
                current_text += " "
            current_text += text
            current_indices.append(idx)
            current_size += text_size
            i += 1
        
        # Add the last chunk if it has content
        if current_text:
            chunk_id = f"text_{current_indices[0]}_{current_indices[-1]}"
            chunk = {
                "id": chunk_id,
                "content": current_text,
                "metadata": {
                    "type": "text",
                    "page_idx": page_idx,
                    "original_indices": current_indices
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def _process_table_items(self, table_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process table items, converting HTML to JSON.
        
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
            
            # Create chunk
            chunk = {
                "id": chunk_id,
                "content": table_text,  # Use text representation as content
                "metadata": {
                    "type": "table",
                    "page_idx": item.get("page_idx", 0),
                    "original_index": item["original_index"],
                    "caption": table_caption,
                    "table_data": table_json  # Store JSON data in metadata
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
    
    def _process_image_items(self, image_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process image items.
        
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
            
            # Create chunk
            chunk = {
                "id": chunk_id,
                "content": content,
                "metadata": {
                    "type": "image",
                    "page_idx": item.get("page_idx", 0),
                    "original_index": item["original_index"],
                    "img_path": img_path,
                    "caption": img_caption
                }
            }
            chunks.append(chunk)
        
        return chunks
    
# Command-line interface for document uploading
def run_uploader_interface():
    """Run the document uploader interface"""
    print("📄 Document Uploader Interface")
    print("============================")
    print("Available commands:")
    print("- 'upload <file_path>': Upload and process a document")
    print("- 'list': List all processed documents")
    print("- 'info <document_id>': Show detailed information about a document")
    print("- 'delete <document_id>': Delete a document")
    print("- 'exit': Exit the interface")
    print("============================")
    
    # Initialize the uploader
    uploader = DocumentUploader()
    
    # Main interaction loop
    while True:
        user_input = input("\n📤 Enter command: ").strip()
        
        if user_input.lower() == 'exit':
            print("Exiting document uploader interface...")
            break
        
        elif user_input.lower().startswith('upload '):
            file_path = user_input[7:].strip()
            
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue
            
            print(f"⏳ Processing document: {file_path} (this may take a while)...")
            result = uploader.upload_document(file_path)
            
            if result["status"] == "success":
                print(f"✅ Document processed successfully!")
                print(f"  - Document ID: {result['document_id']}")
                print(f"  - Processing time: {result['processing_time']:.2f} seconds")
                print(f"  - Chunks: {result['chunks_count']}")
                print(f"  - RAPTOR levels: {', '.join(map(str, result['raptor_levels']))}")
            else:
                print(f"❌ Error processing document: {result['error']}")
        
        elif user_input.lower() == 'list':
            documents = uploader.list_documents()
            
            if not documents:
                print("📋 No documents have been processed yet")
            else:
                print("\n📋 Processed Documents:")
                print("----------------------")
                
                for doc in documents:
                    status_icon = "✅" if doc["status"] == "processed" else "❌"
                    print(f"{status_icon} {doc['document_id']} - {doc['filename']}")
                    print(f"    Date: {doc['processing_date'][:10]}")
                    print(f"    Chunks: {doc['chunks_count']}")
                    
                    # Show content types if available
                    if doc.get("content_types"):
                        types_str = ", ".join([f"{count} {type}" for type, count in doc["content_types"].items()])
                        print(f"    Content: {types_str}")
                    print()
        
        elif user_input.lower().startswith('info '):
            doc_id = user_input[5:].strip()
            metadata = uploader.get_document_metadata(doc_id)
            
            if metadata:
                print(f"\n📄 Document: {doc_id}")
                print("-------------------")
                print(f"Filename: {metadata.get('original_filename', 'Unknown')}")
                print(f"Status: {metadata.get('status', 'Unknown')}")
                print(f"Processing Date: {metadata.get('processing_date', 'Unknown')[:10]}")
                print(f"Processing Time: {metadata.get('processing_time', 0):.2f} seconds")
                print(f"Chunks: {metadata.get('chunks_count', 0)}")
                
                # Show content types breakdown
                if "content_types" in metadata:
                    print("\nContent Types:")
                    for content_type, count in metadata["content_types"].items():
                        print(f"  - {content_type}: {count}")
                
                # Show RAPTOR levels
                if "raptor_levels" in metadata:
                    print("\nRAPTOR Levels:")
                    for level in sorted(metadata["raptor_levels"]):
                        print(f"  - Level {level}")
                
                # Show user metadata if available
                if "user_metadata" in metadata and metadata["user_metadata"]:
                    print("\nUser Metadata:")
                    for key, value in metadata["user_metadata"].items():
                        print(f"  - {key}: {value}")
            else:
                print(f"❌ Document {doc_id} not found")
        
        elif user_input.lower().startswith('delete '):
            doc_id = user_input[7:].strip()
            
            # Ask for confirmation
            confirm = input(f"⚠️ Are you sure you want to delete document {doc_id}? (y/n): ").strip().lower()
            
            if confirm == 'y':
                result = uploader.delete_document(doc_id)
                if result:
                    print(f"✅ Document {doc_id} deleted successfully")
                else:
                    print(f"❌ Error deleting document {doc_id}")
            else:
                print("Deletion cancelled")
        
        else:
            print("❓ Unknown command. Type 'upload <file_path>', 'list', 'info <document_id>', 'delete <document_id>', or 'exit'")


if __name__ == "__main__":
    run_uploader_interface()