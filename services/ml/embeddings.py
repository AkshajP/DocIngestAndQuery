import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating embeddings using Ollama.
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        batch_size: int = 5,
        default_dimension: int = 3072  # Added default dimension
    ):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model
            base_url: Base URL for Ollama API
            batch_size: Maximum batch size for embedding generation
            default_dimension: Default embedding dimension to use if test fails
        """
        self.model_name = model_name
        self.base_url = base_url
        self.batch_size = batch_size
        self.embeddings = None
        self.embedding_dim = default_dimension  # Initialize with default
        
        # Initialize embeddings client
        self._initialize_embeddings()
    
    def _initialize_embeddings(self) -> None:
        """Initialize the embeddings client"""
        try:
            self.embeddings = OllamaEmbeddings(
                model=self.model_name,
                base_url=self.base_url
            )
            
            # Test to get embedding dimension
            try:
                test_embedding = self.embeddings.embed_query("Test embedding dimension")
                if test_embedding and len(test_embedding) > 0:
                    self.embedding_dim = len(test_embedding)
                    logger.info(f"Detected embedding dimension: {self.embedding_dim}")
                else:
                    logger.warning(f"Could not detect embedding dimension, using default: {self.embedding_dim}")
            except Exception as dim_error:
                logger.warning(f"Error detecting embedding dimension: {str(dim_error)}")
                logger.warning(f"Using default dimension: {self.embedding_dim}")
            
            logger.info(f"Initialized embedding service with model {self.model_name}, dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error initializing embedding service: {str(e)}")
            # Continue with the default dimension rather than raising
            logger.warning(f"Using default dimension: {self.embedding_dim}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not self.embeddings:
            self._initialize_embeddings()
        
        try:
            embedding = self.embeddings.embed_query(text)
            actual_dim = len(embedding)
            
            # Ensure we have the correct dimension
            if actual_dim != self.embedding_dim:
                logger.warning(f"Embedding dimension mismatch: got {actual_dim}, expected {self.embedding_dim}")
                # Pad or truncate to correct dimension
                if actual_dim < self.embedding_dim:
                    # Pad with zeros
                    embedding = embedding + [0.0] * (self.embedding_dim - actual_dim)
                else:
                    # Truncate
                    embedding = embedding[:self.embedding_dim]
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dim
    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to log progress information
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        if not self.embeddings:
            self._initialize_embeddings()
        
        all_embeddings = []
        total_texts = len(texts)
        
        # Process in smaller batches to avoid memory issues
        # Reduced batch size for more stability
        effective_batch_size = min(self.batch_size, 5)
        
        for i in range(0, total_texts, effective_batch_size):
            batch_start = time.time()
            
            batch_end = min(i + effective_batch_size, total_texts)
            batch_texts = texts[i:batch_end]
            
            try:
                # Generate embeddings for this batch
                batch_embeddings = []
                # Process each text individually for better error handling
                for text in batch_texts:
                    try:
                        embedding = self.generate_embedding(text)  # Use our improved method
                        batch_embeddings.append(embedding)
                    except Exception as text_error:
                        logger.error(f"Error embedding text: {str(text_error)}")
                        batch_embeddings.append([0.0] * self.embedding_dim)
                
                all_embeddings.extend(batch_embeddings)
                
                if show_progress:
                    batch_time = time.time() - batch_start
                    logger.info(f"Embedded batch {i//effective_batch_size + 1}/{(total_texts-1)//effective_batch_size + 1} "
                               f"({batch_end}/{total_texts}) in {batch_time:.2f}s")
                    
            except Exception as e:
                logger.error(f"Error embedding batch {i//effective_batch_size + 1}: {str(e)}")
                # Use zero vectors as fallback
                zero_vectors = [[0.0] * self.embedding_dim for _ in range(len(batch_texts))]
                all_embeddings.extend(zero_vectors)
        
        # Final verification to ensure all embeddings have the correct dimension
        for i, emb in enumerate(all_embeddings):
            if len(emb) != self.embedding_dim:
                logger.warning(f"Fixing embedding {i} with wrong dimension: {len(emb)} != {self.embedding_dim}")
                if len(emb) < self.embedding_dim:
                    all_embeddings[i] = emb + [0.0] * (self.embedding_dim - len(emb))
                else:
                    all_embeddings[i] = emb[:self.embedding_dim]
        
        return all_embeddings
    
    def generate_embeddings_dict(
        self, 
        texts_dict: Dict[str, str], 
        show_progress: bool = False
    ) -> Dict[str, List[float]]:
        """
        Generate embeddings for texts with associated IDs.
        
        Args:
            texts_dict: Dictionary mapping IDs to texts
            show_progress: Whether to log progress information
            
        Returns:
            Dictionary mapping IDs to embedding vectors
        """
        # Extract IDs and texts
        ids = list(texts_dict.keys())
        texts = list(texts_dict.values())
        
        # Generate embeddings for texts
        embeddings = self.generate_embeddings_batch(texts, show_progress)
        
        # Map embeddings back to IDs
        return {id_: embedding for id_, embedding in zip(ids, embeddings)}