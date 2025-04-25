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
        batch_size: int = 10
    ):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model
            base_url: Base URL for Ollama API
            batch_size: Maximum batch size for embedding generation
        """
        self.model_name = model_name
        self.base_url = base_url
        self.batch_size = batch_size
        self.embeddings = None
        self.embedding_dim = None
        
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
            test_embedding = self.generate_embedding("Test embedding dimension")
            self.embedding_dim = len(test_embedding)
            
            logger.info(f"Initialized embedding service with model {self.model_name}, dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error initializing embedding service: {str(e)}")
            raise
    
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
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * (self.embedding_dim or 768)
    
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
        
        # Process in batches to avoid memory issues
        for i in range(0, total_texts, self.batch_size):
            batch_start = time.time()
            
            batch_end = min(i + self.batch_size, total_texts)
            batch_texts = texts[i:batch_end]
            
            try:
                # Generate embeddings for this batch
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                if show_progress:
                    batch_time = time.time() - batch_start
                    logger.info(f"Embedded batch {i//self.batch_size + 1}/{(total_texts-1)//self.batch_size + 1} "
                               f"({batch_end}/{total_texts}) in {batch_time:.2f}s")
                    
            except Exception as e:
                logger.error(f"Error embedding batch {i//self.batch_size + 1}: {str(e)}")
                # Use zero vectors as fallback
                zero_vectors = [[0.0] * (self.embedding_dim or 768) for _ in range(len(batch_texts))]
                all_embeddings.extend(zero_vectors)
        
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