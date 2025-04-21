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
import boto3
from botocore.exceptions import ClientError

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

class StorageAdapter:
    """Base class for storage operations, abstracting file I/O."""
    
    def write_file(self, content, path):
        """Write content to a file path."""
        pass
        
    def read_file(self, path):
        """Read content from a file path."""
        pass
        
    def delete_file(self, path):
        """Delete a file at the given path."""
        pass
        
    def file_exists(self, path):
        """Check if a file exists at the given path."""
        pass
        
    def create_directory(self, path):
        """Create a directory at the given path."""
        pass
        
    def list_files(self, directory):
        """List files in a directory."""
        pass
        
    def delete_directory(self, path):
        """Delete a directory and its contents."""
        pass
    
class LocalStorageAdapter(StorageAdapter):
    """Local filesystem implementation of StorageAdapter."""
    
    def write_file(self, content, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(content, str):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        elif isinstance(content, bytes):
            with open(path, 'wb') as f:
                f.write(content)
        else:
            raise TypeError("Content must be string or bytes")
    
    def read_file(self, path):
        if not os.path.exists(path):
            return None
        
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def delete_file(self, path):
        if os.path.exists(path):
            os.remove(path)
    
    def file_exists(self, path):
        return os.path.exists(path)
    
    def create_directory(self, path):
        os.makedirs(path, exist_ok=True)
    
    def list_files(self, directory):
        if os.path.exists(directory):
            return os.listdir(directory)
        return []
    
    def delete_directory(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
            
class S3StorageAdapter(StorageAdapter):
    """S3 implementation of StorageAdapter."""
    
    def __init__(self, bucket_name, prefix="", region="us-east-1"):
        self.bucket = bucket_name
        self.prefix = prefix
        self.s3_client = boto3.client('s3', region_name=region)
    
    def _get_full_path(self, path):
        """Convert local path to S3 key."""
        # Remove leading slash if present
        if path.startswith('/'):
            path = path[1:]
        # Add prefix if it exists
        if self.prefix:
            return f"{self.prefix.rstrip('/')}/{path}"
        return path
    
    def write_file(self, content, path):
        s3_key = self._get_full_path(path)
        try:
            if isinstance(content, str):
                self.s3_client.put_object(
                    Bucket=self.bucket,
                    Key=s3_key,
                    Body=content.encode('utf-8')
                )
            elif isinstance(content, bytes):
                self.s3_client.put_object(
                    Bucket=self.bucket,
                    Key=s3_key,
                    Body=content
                )
            else:
                raise TypeError("Content must be string or bytes")
        except ClientError as e:
            logger.error(f"Error writing to S3: {str(e)}")
            raise
    
    def read_file(self, path):
        s3_key = self._get_full_path(path)
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            content = response['Body'].read()
            
            if path.endswith('.pkl'):
                return pickle.loads(content)
            else:
                return content.decode('utf-8')
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            logger.error(f"Error reading from S3: {str(e)}")
            raise
    
    def delete_file(self, path):
        s3_key = self._get_full_path(path)
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
        except ClientError as e:
            logger.error(f"Error deleting from S3: {str(e)}")
            raise
    
    def file_exists(self, path):
        s3_key = self._get_full_path(path)
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logger.error(f"Error checking file in S3: {str(e)}")
            raise
    
    def create_directory(self, path):
        # S3 doesn't need directory creation, but we can add an empty marker
        s3_key = self._get_full_path(path.rstrip('/') + '/.marker')
        try:
            self.s3_client.put_object(Bucket=self.bucket, Key=s3_key, Body=b'')
        except ClientError as e:
            logger.error(f"Error creating S3 directory marker: {str(e)}")
            raise
    
    def list_files(self, directory):
        # Ensure directory ends with a slash
        if not directory.endswith('/'):
            directory += '/'
        
        s3_prefix = self._get_full_path(directory)
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=s3_prefix,
                Delimiter='/'
            )
            
            files = []
            
            # Get files (Contents)
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Extract just the filename from the full path
                    key = obj['Key']
                    file_name = key.split('/')[-1]
                    if file_name and file_name != '.marker':
                        files.append(file_name)
            
            return files
        except ClientError as e:
            logger.error(f"Error listing S3 files: {str(e)}")
            raise
    
    def delete_directory(self, path):
        s3_prefix = self._get_full_path(path.rstrip('/') + '/')
        try:
            # S3 doesn't have directories, so we delete all objects with the prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix)
            
            delete_list = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        delete_list.append({'Key': obj['Key']})
            
            if delete_list:
                # Delete in batches of 1000 (S3 limit)
                for i in range(0, len(delete_list), 1000):
                    batch = delete_list[i:i+1000]
                    self.s3_client.delete_objects(
                        Bucket=self.bucket,
                        Delete={'Objects': batch}
                    )
                    
        except ClientError as e:
            logger.error(f"Error deleting S3 directory: {str(e)}")
            raise


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
        max_tree_levels: int = 3,
        storage_type: str = "local",  # Options: "local" or "s3"
        s3_bucket: str = None,
        s3_prefix: str = "",
        aws_region: str = "us-east-1"
    ):
        """
        Initialize the document uploader.
        
        Args:
            storage_dir: Base directory for storing processed documents (local storage only)
            ollama_base_url: URL for Ollama API
            ollama_model: Model for LLM operations
            ollama_embed_model: Model for embeddings
            language: Language for OCR and processing
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between text chunks
            max_tree_levels: Maximum levels for RAPTOR tree
            storage_type: Storage type ("local" or "s3")
            s3_bucket: S3 bucket name (required if storage_type is "s3")
            s3_prefix: Prefix for S3 keys (like a directory)
            aws_region: AWS region for S3 bucket
        """
        self.storage_dir = storage_dir
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.ollama_embed_model = ollama_embed_model
        self.language = language
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tree_levels = max_tree_levels
        self.storage_type = storage_type
        
        # Initialize storage adapter based on configuration
        if storage_type.lower() == "s3":
            if s3_bucket is None:
                raise ValueError("S3 bucket name is required when using S3 storage")
            logger.info(f"Initializing S3 storage adapter with bucket {s3_bucket}")
            self.storage = S3StorageAdapter(s3_bucket, prefix=s3_prefix, region=aws_region)
            self.registry_path = f"{s3_prefix.rstrip('/')}/document_registry.json"
        else:
            logger.info(f"Initializing local storage adapter with directory {storage_dir}")
            self.storage = LocalStorageAdapter()
            self.storage.create_directory(storage_dir)
            self.registry_path = os.path.join(storage_dir, "document_registry.json")
        
        # Create registry file if it doesn't exist
        if not self.storage.file_exists(self.registry_path):
            self.storage.write_file(
                json.dumps({"documents": {}, "last_updated": datetime.now().isoformat()}, indent=2),
                self.registry_path
            )
        
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
        doc_dir = os.path.join(self.storage_dir, document_id) if self.storage_type == "local" else f"{document_id}"
        
        # Check if document already exists
        if self.storage.file_exists(doc_dir):
            logger.warning(f"Document {document_id} already exists. Using a new ID.")
            document_id = f"doc_{uuid.uuid4().hex[:10]}_{os.path.basename(file_path)}"
            doc_dir = os.path.join(self.storage_dir, document_id) if self.storage_type == "local" else f"{document_id}"
        
        # Create directory structure
        self.storage.create_directory(doc_dir)
        self.storage.create_directory(os.path.join(doc_dir, "chunks"))
        self.storage.create_directory(os.path.join(doc_dir, "embeddings"))
        self.storage.create_directory(os.path.join(doc_dir, "raptor_tree"))
        self.storage.create_directory(os.path.join(doc_dir, "mineru_output"))
        
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
                self.storage.write_file(json.dumps(doc_metadata, indent=2), metadata_path)
                
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
                self.storage.write_file(json.dumps(failure_metadata, indent=2), metadata_path)
                
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
            self.storage.write_file(json.dumps(mineru_output["content_list"], indent=2), content_list_path)
            
            # Save images if present
            if "images" in mineru_output and mineru_output["images"]:
                images_dir = os.path.join(mineru_dir, "images")
                self.storage.create_directory(images_dir)
                
                for img_name, img_data in mineru_output["images"].items():
                    img_path = os.path.join(images_dir, img_name)
                    self.storage.write_file(img_data, img_path)  # Binary data
            
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
            self.storage.write_file(json.dumps(chunk, indent=2), chunk_path)
        
        # Save all chunks as a single file for convenience
        all_chunks_path = os.path.join(doc_dir, "all_chunks.json")
        self.storage.write_file(json.dumps(all_chunks, indent=2), all_chunks_path)
        
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
                self.storage.write_file(pickle.dumps(emb), emb_path)
        
        # Save all embeddings as a single file
        all_embeddings_path = os.path.join(doc_dir, "all_embeddings.pkl")
        self.storage.write_file(pickle.dumps(all_embeddings), all_embeddings_path)
        
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
            self.storage.write_file(pickle.dumps(df_layer0), layer0_path)
            
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
                    self.storage.write_file(pickle.dumps(df_clusters), clusters_path)
                    
                    summary_path = os.path.join(raptor_dir, f"layer{current_level}_summaries.pkl")
                    self.storage.write_file(pickle.dumps(df_summary), summary_path)
                    
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
                    
                    self.storage.write_file(json.dumps(clusters_json, indent=2), clusters_json_path)
                    self.storage.write_file(json.dumps(summary_json, indent=2), summary_json_path)
                    
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
            self.storage.write_file(json.dumps(tree_structure, indent=2), tree_structure_path)
            
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
        registry_content = self.storage.read_file(self.registry_path)
        registry = json.loads(registry_content) if registry_content else {"documents": {}, "last_updated": datetime.now().isoformat()}
        
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
        self.storage.write_file(json.dumps(registry, indent=2), self.registry_path)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the registry.
        
        Returns:
            List of document metadata
        """
        # Load registry
        registry_content = self.storage.read_file(self.registry_path)
        registry = json.loads(registry_content) if registry_content else {"documents": {}, "last_updated": datetime.now().isoformat()}
        
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
        doc_dir = os.path.join(self.storage_dir, document_id) if self.storage_type == "local" else f"{document_id}"
        metadata_path = os.path.join(doc_dir, "metadata.json")
        
        metadata_content = self.storage.read_file(metadata_path)
        if metadata_content:
            return json.loads(metadata_content)
        
        return None
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its processed data.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        doc_dir = os.path.join(self.storage_dir, document_id) if self.storage_type == "local" else f"{document_id}"
        
        if not self.storage.file_exists(doc_dir):
            logger.warning(f"Document {document_id} not found")
            return False
        
        try:
            # Remove from registry first
            registry_content = self.storage.read_file(self.registry_path)
            registry = json.loads(registry_content) if registry_content else {"documents": {}, "last_updated": datetime.now().isoformat()}
            
            if document_id in registry["documents"]:
                del registry["documents"][document_id]
                registry["last_updated"] = datetime.now().isoformat()
                
                self.storage.write_file(json.dumps(registry, indent=2), self.registry_path)
            
            # Delete document directory
            self.storage.delete_directory(doc_dir)
            
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
    print("- 'config': Show current storage configuration")
    print("- 'exit': Exit the interface")
    print("============================")
    
    # Get storage configuration from environment variables
    storage_type = os.environ.get("STORAGE_TYPE", "local")
    s3_bucket = os.environ.get("S3_BUCKET", None)
    s3_prefix = os.environ.get("S3_PREFIX", "")
    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    
    # Print storage configuration
    print("\n🔧 Storage Configuration:")
    print(f"- Storage Type: {storage_type}")
    if storage_type.lower() == "s3":
        print(f"- S3 Bucket: {s3_bucket}")
        print(f"- S3 Prefix: {s3_prefix}")
        print(f"- AWS Region: {aws_region}")
    else:
        print(f"- Local Storage Directory: document_store")
    print()
    
    # Initialize the uploader with the appropriate storage config
    if storage_type.lower() == "s3":
        if not s3_bucket:
            print("❌ S3 bucket name is required when using S3 storage.")
            print("Please set the S3_BUCKET environment variable.")
            return
        
        uploader = DocumentUploader(
            storage_type="s3",
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            aws_region=aws_region
        )
    else:
        uploader = DocumentUploader()
    
    # Main interaction loop
    while True:
        user_input = input("\n📤 Enter command: ").strip()
        
        if user_input.lower() == 'exit':
            print("Exiting document uploader interface...")
            break
        
        elif user_input.lower() == 'config':
            print("\n🔧 Current Storage Configuration:")
            print(f"- Storage Type: {uploader.storage_type}")
            if uploader.storage_type.lower() == "s3":
                print(f"- S3 Bucket: {s3_bucket}")
                print(f"- S3 Prefix: {s3_prefix}")
                print(f"- AWS Region: {aws_region}")
            else:
                print(f"- Local Storage Directory: {uploader.storage_dir}")
        
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
            print("❓ Unknown command. Type 'upload <file_path>', 'list', 'info <document_id>', 'delete <document_id>', 'config', or 'exit'")

if __name__ == "__main__":
    run_uploader_interface()