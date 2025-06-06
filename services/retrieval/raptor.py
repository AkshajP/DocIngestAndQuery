import logging
import numpy as np
import pandas as pd
import math
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.exceptions import ConvergenceWarning
import warnings
from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM

class Raptor:
    """
    Enhanced Retrieval Augmented Processing and Tree-based Optimization for Retrieval.
    
    Builds a hierarchical tree representation of document chunks with adaptive
    clustering strategies based on document characteristics. Automatically
    adjusts to handle documents of all sizes, including those with too few chunks.
    """
    
    def __init__(
        self,
        ollama_base_url: str,
        ollama_model: str,
        ollama_embed_model: str,
        max_tree_levels: int = 3,
        min_chunks_per_cluster: int = 3,
        clustering_methods: List[str] = None,
        min_chunks_for_tree: int = 10,
        embed_dim: int = 3072,
        summary_temperature: float = 0.3
    ):
        """
        Initialize the RAPTOR tree builder.
        
        Args:
            ollama_base_url: Base URL for Ollama API
            ollama_model: LLM model name for summarization
            ollama_embed_model: Embedding model name
            max_tree_levels: Maximum tree depth
            min_chunks_per_cluster: Minimum chunks per cluster
            clustering_methods: List of clustering methods to try in order of preference
            min_chunks_for_tree: Minimum number of chunks required for hierarchical tree
            embed_dim: Embedding dimension
            summary_temperature: Temperature for summary generation
        """
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.ollama_embed_model = ollama_embed_model
        self.max_tree_levels = max_tree_levels
        self.min_chunks_per_cluster = min_chunks_per_cluster
        self.min_chunks_for_tree = min_chunks_for_tree
        self.embed_dim = embed_dim
        self.summary_temperature = summary_temperature
        
        # Default clustering methods if not provided
        if clustering_methods is None:
            self.clustering_methods = ["spectral", "kmeans", "agglomerative", "flat"]
        else:
            self.clustering_methods = clustering_methods
            
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM for summarization
        self._init_llm()
    
    def _init_llm(self):
        """
        Initialize the LLM for generating summaries with one fallback option.
        Tries to initialize with the preferred model, falls back to alternative if needed.
        """
        # Try to initialize the LLM with one fallback
        models_to_try = [
            self.ollama_model,  # Primary model
            "llama3.2"          # Fallback model
        ]
        
        for model in models_to_try:
            try:
                self.logger.info(f"Attempting to initialize LLM with model: {model}")
                self.llm = OllamaLLM(
                    model=model,
                    base_url=self.ollama_base_url,
                    temperature=self.summary_temperature
                )
                
                # Test the model with a simple prompt
                test_prompt = "Respond with 'ok' if you can understand this."
                try:
                    test_response = self.llm.invoke(test_prompt)
                    if test_response and len(test_response) > 0:
                        self.logger.info(f"Successfully initialized LLM with model {model}")
                        return
                    else:
                        self.logger.warning(f"Model {model} initialized but returned empty response")
                except Exception as test_error:
                    self.logger.warning(f"Test invocation failed for model {model}: {str(test_error)}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM with model {model}: {str(e)}")
                continue
        
        # If we get here, all models failed
        self.logger.error("All LLM initialization attempts failed")
        self.llm = None
    
    def build_tree(
        self, 
        chunk_texts: List[str], 
        chunk_ids: List[str], 
        chunk_embeddings: Dict[str, List[float]],
        content_types: Optional[List[str]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Build a hierarchical tree representation of document chunks.
        Adaptively selects clustering approach based on document characteristics.
        
        Args:
            chunk_texts: List of chunk text content
            chunk_ids: List of corresponding chunk IDs
            chunk_embeddings: Dictionary mapping chunk IDs to embeddings
            content_types: Optional list of content types for each chunk
            
        Returns:
            Dictionary with tree structure data at each level
        """
        start_time = time.time()
        
        # Get number of chunks
        n_chunks = len(chunk_texts)
        self.logger.info(f"Building tree for {n_chunks} chunks")
        
        # Validate inputs with robust error handling
        if n_chunks == 0:
            self.logger.warning("No chunks provided, cannot build tree")
            return {0: {"texts": [], "ids": []}}
        
        if n_chunks != len(chunk_ids):
            self.logger.error(f"Mismatch between chunk_texts ({n_chunks}) and chunk_ids ({len(chunk_ids)})")
            # Try to salvage by trimming to the shorter length
            min_length = min(n_chunks, len(chunk_ids))
            chunk_texts = chunk_texts[:min_length]
            chunk_ids = chunk_ids[:min_length]
            n_chunks = min_length
            self.logger.info(f"Adjusted to use {n_chunks} chunks")
            
        # Verify all chunk_ids exist in chunk_embeddings
        missing_embeddings = [chunk_id for chunk_id in chunk_ids if chunk_id not in chunk_embeddings]
        if missing_embeddings:
            self.logger.warning(f"Missing embeddings for {len(missing_embeddings)} chunks")
            # Filter out chunks without embeddings
            valid_indices = [i for i, chunk_id in enumerate(chunk_ids) if chunk_id in chunk_embeddings]
            chunk_texts = [chunk_texts[i] for i in valid_indices]
            chunk_ids = [chunk_ids[i] for i in valid_indices]
            n_chunks = len(chunk_ids)
            self.logger.info(f"Adjusted to use {n_chunks} chunks with embeddings")
            
        # Handle edge cases with very few chunks
        if n_chunks == 0:
            self.logger.error("No valid chunks with embeddings remaining")
            return {0: {"texts": [], "ids": []}}
            
        if n_chunks == 1:
            self.logger.info("Document has only 1 chunk, using single-level structure")
            return {0: {"texts": chunk_texts, "ids": chunk_ids}}
        
        # If too few chunks for hierarchical clustering, use flat approach
        if n_chunks < self.min_chunks_for_tree:
            self.logger.info(f"Document has only {n_chunks} chunks (< {self.min_chunks_for_tree}), using flat organization")
            return self._build_flat_structure(chunk_texts, chunk_ids, chunk_embeddings)
        
        # Dynamically adjust tree levels based on chunk count
        # We want at least min_chunks_per_cluster chunks at the lowest level
        try:
            # Safely calculate log with a minimum value of 1
            log_value = max(1, math.log(max(2, n_chunks / self.min_chunks_per_cluster), 2))
            adjusted_max_levels = min(self.max_tree_levels, max(1, int(log_value)))
        except Exception as e:
            self.logger.warning(f"Error calculating adjusted levels: {str(e)}")
            adjusted_max_levels = min(self.max_tree_levels, max(1, n_chunks // 10))
            
        self.logger.info(f"Using adjusted tree depth: {adjusted_max_levels} (original: {self.max_tree_levels})")
        
        try:
            # Create embedding matrix from the dictionary
            # Make sure to preserve order matching chunk_ids
            embedding_list = [chunk_embeddings[chunk_id] for chunk_id in chunk_ids]
            embedding_matrix = np.array(embedding_list)
            
            # Check embeddings for NaN or Inf values
            if np.isnan(embedding_matrix).any() or np.isinf(embedding_matrix).any():
                self.logger.warning("Embeddings contain NaN or Inf values, replacing with zeros")
                embedding_matrix = np.nan_to_num(embedding_matrix)
            
            # Check if embeddings have the expected dimension
            if embedding_matrix.shape[1] != self.embed_dim:
                self.logger.warning(f"Embedding dimension mismatch: got {embedding_matrix.shape[1]}, expected {self.embed_dim}")
                
                # Try to fix by padding or truncating
                if embedding_matrix.shape[1] < self.embed_dim:
                    # Pad with zeros
                    padding = np.zeros((embedding_matrix.shape[0], self.embed_dim - embedding_matrix.shape[1]))
                    embedding_matrix = np.hstack((embedding_matrix, padding))
                    self.logger.info(f"Padded embeddings to {self.embed_dim} dimensions")
                else:
                    # Truncate
                    embedding_matrix = embedding_matrix[:, :self.embed_dim]
                    self.logger.info(f"Truncated embeddings to {self.embed_dim} dimensions")
                    
            # Build tree with adaptive clustering
            tree_data = self._build_adaptive_tree(
                chunk_texts, 
                chunk_ids, 
                embedding_matrix, 
                adjusted_max_levels, 
                content_types
            )
            
        except Exception as e:
            self.logger.error(f"Error in tree building process: {str(e)}")
            # Fall back to flat structure in case of any error
            self.logger.info("Falling back to flat structure due to error")
            tree_data = self._build_flat_structure(chunk_texts, chunk_ids, chunk_embeddings)
        
        build_time = time.time() - start_time
        self.logger.info(f"Tree building completed in {build_time:.2f} seconds with {len(tree_data)} levels")
        
        return tree_data
    
    def _build_adaptive_tree(
        self, 
        chunk_texts: List[str], 
        chunk_ids: List[str], 
        embedding_matrix: np.ndarray, 
        max_levels: int,
        content_types: Optional[List[str]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Build tree structure using the most appropriate clustering method.
        Tries multiple methods in order of preference until one succeeds.
        
        Args:
            chunk_texts: List of chunk text content
            chunk_ids: List of chunk IDs
            embedding_matrix: Matrix of chunk embeddings
            max_levels: Maximum number of tree levels to build
            content_types: Optional list of content types for filtering
            
        Returns:
            Dictionary with tree structure data at each level
        """
        # Initialize tree data with level 0 (original chunks)
        tree_data = {0: {"texts": chunk_texts, "ids": chunk_ids}}
        
        # If content types are provided, add them to level 0
        if content_types:
            tree_data[0]["content_types"] = content_types
        
        # Try building each level
        for level in range(1, max_levels + 1):
            # For each level, try different clustering methods until one works
            clustering_success = False
            
            for method in self.clustering_methods:
                try:
                    self.logger.info(f"Attempting level {level} clustering with method: {method}")
                    
                    level_data = self._apply_clustering_method(
                        method, 
                        tree_data[level-1]["texts"],
                        tree_data[level-1]["ids"],
                        embedding_matrix,
                        level
                    )
                    
                    # If we got valid results, store them and move to next level
                    if level_data and "clusters" in level_data and "summaries_df" in level_data:
                        tree_data[level] = level_data
                        clustering_success = True
                        self.logger.info(f"Successfully built level {level} using {method}")
                        break
                        
                except Exception as e:
                    self.logger.warning(f"Clustering method {method} failed for level {level}: {str(e)}")
                    continue
            
            # If all clustering methods failed for this level, stop building the tree
            if not clustering_success:
                self.logger.warning(f"Could not build level {level} with any method. Tree building stopped at level {level-1}.")
                break
                
        return tree_data
    
    def _apply_clustering_method(
        self, 
        method: str, 
        texts: List[str], 
        ids: List[str], 
        embedding_matrix: np.ndarray, 
        level: int
    ) -> Optional[Dict[str, Any]]:
        """
        Apply a specific clustering method and handle errors gracefully.
        
        Args:
            method: Clustering method name
            texts: List of texts to cluster
            ids: List of corresponding IDs
            embedding_matrix: Matrix of embeddings
            level: Current tree level
            
        Returns:
            Dictionary with clustering results or None if method failed
        """
        if method == "spectral":
            return self._apply_spectral_clustering(texts, ids, embedding_matrix, level)
        elif method == "kmeans":
            return self._apply_kmeans_clustering(texts, ids, embedding_matrix, level)
        elif method == "agglomerative":
            return self._apply_agglomerative_clustering(texts, ids, embedding_matrix, level)
        elif method == "flat":
            return self._apply_flat_organization(texts, ids, level)
        else:
            self.logger.warning(f"Unknown clustering method: {method}")
            return None
    
    def _apply_spectral_clustering(
        self, 
        texts: List[str], 
        ids: List[str], 
        embedding_matrix: np.ndarray, 
        level: int
    ) -> Optional[Dict[str, Any]]:
        """
        Apply spectral clustering with progressive fallback strategies.
        
        Args:
            texts: List of texts to cluster
            ids: List of corresponding IDs
            embedding_matrix: Matrix of embeddings
            level: Current tree level
            
        Returns:
            Dictionary with clustering results or None if all strategies failed
        """
        n_samples = len(texts)
        
        # Define two fallback configurations to try
        fallback_configs = [
            # Start with standard nearest_neighbors approach
            {
                "n_clusters": min(max(2, int(n_samples / self.min_chunks_per_cluster)), max(2, n_samples - 1)),
                "affinity": "nearest_neighbors",
                "n_neighbors": min(15, max(2, n_samples // 2))
            },
            # Fallback to RBF kernel (dense approach)
            {
                "n_clusters": min(max(2, int(n_samples / self.min_chunks_per_cluster)), max(2, n_samples - 1)),
                "affinity": "rbf",
                "n_neighbors": None
            }
        ]
        
        # Try each configuration until one works
        for i, config in enumerate(fallback_configs):
            try:
                self.logger.info(f"Spectral clustering attempt {i+1}/{len(fallback_configs)}: " + 
                                f"clusters={config['n_clusters']}, affinity={config['affinity']}")
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    
                    # Prepare clustering args based on config
                    clustering_args = {
                        "n_clusters": config["n_clusters"],
                        "affinity": config["affinity"],
                        "random_state": 42
                    }
                    
                    # Add n_neighbors only for nearest_neighbors affinity
                    if config["affinity"] == "nearest_neighbors" and config["n_neighbors"] is not None:
                        clustering_args["n_neighbors"] = config["n_neighbors"]
                    
                    cluster_labels = SpectralClustering(**clustering_args).fit_predict(embedding_matrix)
                    
                # If we get here, the clustering succeeded
                self.logger.info(f"Spectral clustering succeeded with configuration {i+1}")
                return self._process_clustering_results(texts, ids, cluster_labels, level, config["n_clusters"])
                
            except Exception as e:
                self.logger.warning(f"Spectral clustering failed with configuration {i+1}: {str(e)}")
                continue
        
        # If we get here, all fallback configurations failed
        self.logger.error("All spectral clustering configurations failed")
        return None
    
    def _apply_kmeans_clustering(
        self, 
        texts: List[str], 
        ids: List[str], 
        embedding_matrix: np.ndarray, 
        level: int
    ) -> Optional[Dict[str, Any]]:
        """
        Apply KMeans clustering with one fallback strategy.
        
        Args:
            texts: List of texts to cluster
            ids: List of corresponding IDs
            embedding_matrix: Matrix of embeddings
            level: Current tree level
            
        Returns:
            Dictionary with clustering results or None if all strategies failed
        """
        n_samples = len(texts)
        
        # Define two fallback configurations to try
        fallback_configs = [
            # Standard approach
            {
                "n_clusters": min(max(2, int(n_samples / self.min_chunks_per_cluster)), max(2, n_samples - 1)),
                "n_init": 10,
                "max_iter": 300
            },
            # Fallback with fewer clusters, more initializations
            {
                "n_clusters": min(max(2, int(n_samples / (self.min_chunks_per_cluster * 2))), max(2, n_samples - 1)),
                "n_init": 20,
                "max_iter": 500
            }
        ]
        
        # Try each configuration until one works
        for i, config in enumerate(fallback_configs):
            try:
                self.logger.info(f"KMeans clustering attempt {i+1}/{len(fallback_configs)}: " + 
                                f"clusters={config['n_clusters']}, n_init={config['n_init']}")
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    
                    kmeans = KMeans(
                        n_clusters=config["n_clusters"],
                        n_init=config["n_init"],
                        max_iter=config["max_iter"],
                        random_state=42
                    )
                    cluster_labels = kmeans.fit_predict(embedding_matrix)
                
                # If we get here, the clustering succeeded
                self.logger.info(f"KMeans clustering succeeded with configuration {i+1}")
                return self._process_clustering_results(texts, ids, cluster_labels, level, config["n_clusters"])
                
            except Exception as e:
                self.logger.warning(f"KMeans clustering failed with configuration {i+1}: {str(e)}")
                continue
        
        # If we get here, all fallback configurations failed
        self.logger.error("All KMeans clustering configurations failed")
        return None
    
    def _apply_agglomerative_clustering(
        self, 
        texts: List[str], 
        ids: List[str], 
        embedding_matrix: np.ndarray, 
        level: int
    ) -> Optional[Dict[str, Any]]:
        """
        Apply Agglomerative (hierarchical) clustering with one fallback strategy.
        Works well for very small datasets.
        
        Args:
            texts: List of texts to cluster
            ids: List of corresponding IDs
            embedding_matrix: Matrix of embeddings
            level: Current tree level
            
        Returns:
            Dictionary with clustering results or None if all strategies failed
        """
        n_samples = len(texts)
        
        # Define two fallback configurations to try
        fallback_configs = [
            # Standard approach with ward linkage
            {
                "n_clusters": min(max(2, int(n_samples / 2)), max(2, n_samples - 1)),
                "affinity": "euclidean",
                "linkage": "ward"
            },
            # Fallback with average linkage
            {
                "n_clusters": min(max(2, int(n_samples / 2)), max(2, n_samples - 1)),
                "affinity": "euclidean",
                "linkage": "average"
            }
        ]
        
        # Try each configuration until one works
        for i, config in enumerate(fallback_configs):
            try:
                self.logger.info(f"Agglomerative clustering attempt {i+1}/{len(fallback_configs)}: " + 
                                f"clusters={config['n_clusters']}, linkage={config['linkage']}")
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    
                    clustering = AgglomerativeClustering(
                        n_clusters=config["n_clusters"],
                        affinity=config["affinity"],
                        linkage=config["linkage"]
                    )
                    cluster_labels = clustering.fit_predict(embedding_matrix)
                
                # If we get here, the clustering succeeded
                self.logger.info(f"Agglomerative clustering succeeded with configuration {i+1}")
                return self._process_clustering_results(texts, ids, cluster_labels, level, config["n_clusters"])
                
            except Exception as e:
                self.logger.warning(f"Agglomerative clustering failed with configuration {i+1}: {str(e)}")
                continue
        
        # If we get here, all fallback configurations failed
        self.logger.error("All Agglomerative clustering configurations failed")
        return None
    
    def _apply_flat_organization(self, texts, ids, level):
        """
        Create a flat organization of texts as a fallback method.
        Simply groups all texts into a single cluster.
        
        Args:
            texts: List of texts to organize
            ids: List of corresponding IDs
            level: Current tree level
            
        Returns:
            Dictionary with clustering results
        """
        n_samples = len(texts)
        self.logger.info(f"Using flat organization for {n_samples} samples")
        
        # Create a single cluster containing all texts
        cluster_labels = np.zeros(n_samples, dtype=int)
        
        # Process results (summarize the single cluster)
        return self._process_clustering_results(texts, ids, cluster_labels, level, 1)
        
    def _build_flat_structure(self, chunk_texts, chunk_ids, chunk_embeddings):
        """
        Build a flat organization when hierarchical clustering isn't feasible.
        This is used when documents have very few chunks.
        
        Args:
            chunk_texts: List of chunk text content
            chunk_ids: List of chunk IDs
            chunk_embeddings: Dictionary of embeddings
            
        Returns:
            Dictionary with tree structure data
        """
        # We still provide a tree-like structure but with just two levels
        # Level 0: All original chunks
        # Level 1: A single summary of all chunks
        
        tree_data = {
            0: {
                "texts": chunk_texts,
                "ids": chunk_ids
            }
        }
        
        # If we have at least 2 chunks, create a simple summary
        if len(chunk_texts) >= 2:
            try:
                # Create a summary of all chunks
                combined_text = " ".join(chunk_texts)
                summary = self._generate_summary(combined_text)
                
                tree_data[1] = {
                    "texts": [summary],
                    "ids": ["summary_all"],
                    "clusters": [0] * len(chunk_texts),  # All chunks belong to the same cluster
                    "summaries_df": pd.DataFrame({
                        "cluster": [0],
                        "summaries": [summary]
                    })
                }
            except Exception as e:
                self.logger.warning(f"Failed to create summary for flat structure: {str(e)}")
                # Still return the level 0 data even if summarization fails
        
        return tree_data
    
    def _process_clustering_results(self, texts, ids, cluster_labels, level, n_clusters):
        """
        Process clustering results to create level data structure.
        Generates summaries for each cluster with robust error handling.
        
        Args:
            texts: List of texts to cluster
            ids: List of corresponding IDs  
            cluster_labels: Array of cluster assignments
            level: Current tree level
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with processed clustering results
        """
        # Group texts by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((texts[i], ids[i]))
        
        # Generate summaries for each cluster with error handling
        summaries = []
        for cluster_id, cluster_items in clusters.items():
            cluster_texts = [item[0] for item in cluster_items]
            
            try:
                # Skip clusters that are too small
                if len(cluster_texts) < 2:
                    summary = cluster_texts[0]  # Use the single text as its own summary
                else:
                    # Try to generate a summary
                    combined_text = " ".join(cluster_texts)
                    summary = self._generate_summary(combined_text)
                    
                    # Verify summary isn't empty
                    if not summary or len(summary.strip()) == 0:
                        self.logger.warning(f"Empty summary generated for cluster {cluster_id}, using fallback")
                        # Fallback to first few sentences if summary generation failed
                        from nltk.tokenize import sent_tokenize
                        try:
                            sentences = sent_tokenize(combined_text)
                            summary = " ".join(sentences[:min(3, len(sentences))])
                        except:
                            # If sentence tokenization fails, just use the first 200 chars
                            summary = combined_text[:200] + "..."
            except Exception as e:
                self.logger.warning(f"Error generating summary for cluster {cluster_id}: {str(e)}")
                # Fallback to using first text in cluster
                summary = cluster_texts[0][:200] + "..."
                
            summaries.append({
                "cluster": cluster_id,
                "summaries": summary,
                "chunk_count": len(cluster_texts)
            })
        
        # Create DataFrame for summaries
        summaries_df = pd.DataFrame(summaries)
        
        return {
            "clusters": cluster_labels.tolist(),
            "summaries_df": summaries_df.to_dict('records'),
            "n_clusters": n_clusters
        }
    
    def _generate_summary(self, combined_text: str) -> str:
        """
        Generate a summary of the given text with robust error handling.
        
        Args:
            combined_text: Text to summarize
            
        Returns:
            Summary of the text, or a fallback if summarization fails
        """
        if not self.llm:
            self._init_llm()
            
        if not self.llm:
            self.logger.warning("No LLM available for summarization, using fallback")
            # Return first few sentences as fallback
            return combined_text[:300] + "..."
            
        # Trim if text is too long
        max_input_length = 8000
        if len(combined_text) > max_input_length:
            self.logger.info(f"Text too long ({len(combined_text)} chars), truncating to {max_input_length}")
            combined_text = combined_text[:max_input_length] + "..."
            
        try:
            # Define system prompt for summarization
            system_prompt = """
            You are an extractive summarization assistant in a hierarchical summarization system.
            Your task is to extract and compile the most important sentences and factual statements from the input text to produce a dense, information-rich summary.

            Guidelines:
            Preserve as much key information, terminology, and technical detail as possible.
            Select and stitch together original sentences or phrases from the input without paraphrasing.
            Maintain the original meaning, context, and logical flow.
            Prioritize completeness and fidelity to the source over brevity. (change of text to proper case is allowed)
            Do not include commentary, explanations, or reference yourself or the task.
            Respond with just the summary.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Please summarize the following text:\n\n{combined_text}")
            ]
            
            # Set timeout to avoid hanging
            start_time = time.time()
            timeout = 30  # seconds
            
            # Try to generate summary with timeout handling
            summary = None
            try:
                # Use invoke for a simple call
                summary = self.llm.invoke(messages)
            except Exception as e:
                self.logger.warning(f"Error generating summary: {str(e)}")
                summary = None
                
            # Check if we have a valid summary
            if summary and isinstance(summary, str) and len(summary.strip()) > 10:
                return summary.strip()
                
            # If we get here, summarization failed
            self.logger.warning("Generated summary was invalid or too short, using fallback")
            
            # Simple extractive summarization as fallback
            sentences = combined_text.split('. ')
            if len(sentences) <= 3:
                return combined_text
            else:
                return '. '.join(sentences[:3]) + '.'
                
        except Exception as e:
            self.logger.error(f"Exception during summarization: {str(e)}")
            traceback.print_exc()
            # Return first part of text as fallback
            return combined_text[:300] + "..."