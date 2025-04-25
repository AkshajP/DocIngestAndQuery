import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import umap
from sklearn.mixture import GaussianMixture

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class Raptor:
    """
    RAPTOR (Recursive Abstractive Processing for Tree Organized Retrieval) implementation.
    Focused on hierarchical tree building and summarization without vector storage.
    """
    
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2",
        ollama_embed_model: str = "llama3.2",
        max_tree_levels: int = 3,
        random_seed: int = 224
    ):
        """Initialize the RAPTOR system with required components."""
        logger.info("Initializing RAPTOR tree builder")
        self.random_seed = random_seed
        self.max_tree_levels = max_tree_levels
        
        # LLM for summarization
        self.llm = OllamaLLM(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.1,
            max_tokens=1000
        )
        
        # Summarization prompt
        self.summary_prompt = PromptTemplate(
            template="""
            Create a comprehensive and detailed summary of the following document sections.
            Each section is marked with a header indicating its type and source. 
            Prioritise correct information such that it can be used effectively by downstream systems.
            
            Important guidelines:
            1. Maintain key information, important details, and overall context
            2. Pay special attention to TEXT sections which contain the primary content
            3. Include important information from TABLE sections in a structured format
            4. Incorporate previous SUMMARY sections to maintain hierarchical knowledge
            5. Focus on factual information and key concepts
            6. Ensure the summary is coherent, well-structured, and preserves the relationships between concepts
            7. If there is no usable content like only a structure for a blank table or image names, etc then return only the content that you think will be useful for further processing (for ex: just say blank table or image name mentioned)
            
            Do not include any additional commentary or explanations beyond what's requested above or refer to the context in the answer answer.
            
            Document Sections:
            {context}
            
            Summary:
            """,
            input_variables=["context"]
        )
        
        logger.info("RAPTOR initialization complete")
    
    def build_tree(
        self, 
        texts: List[str], 
        chunk_ids: List[str],
        embeddings: Dict[str, List[float]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Build RAPTOR tree from texts and embeddings.
        
        Args:
            texts: List of text chunks
            chunk_ids: List of chunk IDs corresponding to texts
            embeddings: Dictionary mapping chunk IDs to embeddings
            
        Returns:
            Dictionary with tree structure at each level
        """
        logger.info(f"Building RAPTOR tree with {len(texts)} chunks")
        
        if not texts or len(texts) != len(chunk_ids):
            logger.error("Invalid input: texts and chunk_ids must be non-empty and of same length")
            return {}
            
        # Create embedding array from dictionary
        embedding_list = [embeddings[chunk_id] for chunk_id in chunk_ids]
        embedding_array = np.array(embedding_list)
        
        # Build tree recursively
        tree_results = self._recursive_embed_cluster_summarize(
            texts, 
            chunk_ids,
            embedding_array, 
            level=1
        )
        
        logger.info(f"RAPTOR tree construction complete with {len(tree_results)} levels")
        return tree_results
    
    def _recursive_embed_cluster_summarize(
        self,
        texts: List[str],
        chunk_ids: List[str],
        embeddings: np.ndarray,
        level: int = 1
    ) -> Dict[int, Dict[str, Any]]:
        """
        Recursively build tree levels through clustering and summarization.
        
        Args:
            texts: Text chunks to process
            chunk_ids: IDs for each text chunk
            embeddings: Numpy array of embeddings
            level: Current tree level
            
        Returns:
            Dictionary mapping levels to cluster and summary data
        """
        results = {}
        
        # Process current level
        clusters_df, summaries_df = self._embed_cluster_summarize_texts(
            texts, 
            chunk_ids,
            embeddings, 
            level
        )
        
        # Store results
        results[level] = {
            "clusters_df": clusters_df,
            "summaries_df": summaries_df
        }
        
        # Modified condition: Continue building tree as long as we haven't reached max_level
        # and we have at least one text to summarize
        if level < self.max_tree_levels and len(texts) > 0:
            # Use summaries as input for next level
            new_texts = summaries_df["summaries"].tolist()
            new_chunk_ids = [f"summary_l{level}_c{c}" for c in summaries_df["cluster"].tolist()]
            
            # Generate embeddings for summaries - in practice these would be generated 
            # by the embedding service, but we'll use a placeholder here
            next_level_results = self._recursive_embed_cluster_summarize(
                new_texts, 
                new_chunk_ids,
                np.random.randn(len(new_texts), embeddings.shape[1]), 
                level + 1
            )
            
            # Merge results
            results.update(next_level_results)
        
        return results
    
    def _embed_cluster_summarize_texts(
        self,
        texts: List[str],
        chunk_ids: List[str],
        embeddings: np.ndarray,
        level: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Cluster and summarize texts at a specific level.
        
        Args:
            texts: List of texts to process
            chunk_ids: IDs for each text
            embeddings: Numpy array of embeddings
            level: Current tree level
            
        Returns:
            Tuple of (clusters DataFrame, summaries DataFrame)
        """
        logger.info(f"Processing level {level} with {len(texts)} texts")
        
        # Handle edge case with very few texts
        if len(texts) <= 1:
            # For single text, create a simple "cluster"
            df_clusters = pd.DataFrame({
                "text": texts,
                "chunk_id": chunk_ids,
                "cluster": [[0]]  # Single cluster
            })
            
            # Generate a summary
            formatted_txt = f"--- LEVEL {level-1} TEXT ---\n\n{texts[0]}"
            try:
                summary_chain = self.summary_prompt | self.llm | StrOutputParser()
                summary = summary_chain.invoke({"context": formatted_txt})
                logger.debug(f"Generated summary for single text at level {level}")
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}")
                summary = f"Summary of: {texts[0][:100]}..."  # Fallback
            
            # Create summary DataFrame
            df_summary = pd.DataFrame({
                "summaries": [summary],
                "level": [level],
                "cluster": [0]
            })
            
            return df_clusters, df_summary
        
        # Original clustering and summarization for multiple texts
        df_clusters = self._cluster_texts(texts, chunk_ids, embeddings)
        
        # Get unique cluster identifiers
        all_clusters = df_clusters["cluster"].explode().unique()
        
        logger.info(f"Generated {len(all_clusters)} clusters at level {level}")
        
        # Format text within each cluster and generate summaries
        summaries = []
        cluster_ids = []
        
        for cluster_id in all_clusters:
            # Get texts in this cluster
            cluster_mask = df_clusters["cluster"].apply(lambda x: cluster_id in x)
            cluster_texts = df_clusters.loc[cluster_mask, "text"].tolist()
            
            if not cluster_texts:
                continue
                
            # Format texts for summarization
            formatted_txt = self._format_texts_for_summary(cluster_texts, level)
            
            try:
                # Generate summary
                summary_chain = self.summary_prompt | self.llm | StrOutputParser()
                summary = summary_chain.invoke({"context": formatted_txt})
                summaries.append(summary)
                cluster_ids.append(cluster_id)
                logger.debug(f"Generated summary for cluster {cluster_id} at level {level}")
            except Exception as e:
                logger.error(f"Error generating summary for cluster {cluster_id}: {str(e)}")
                # Use a simple concatenation as fallback
                summary = f"Summary of cluster {cluster_id}: " + " ".join(t[:50] + "..." for t in cluster_texts[:3])
                summaries.append(summary)
                cluster_ids.append(cluster_id)
        
        # Create DataFrame for summaries
        df_summary = pd.DataFrame({
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": cluster_ids
        })
        
        return df_clusters, df_summary
    
    def _cluster_texts(
        self, 
        texts: List[str],
        chunk_ids: List[str],
        embeddings: np.ndarray
    ) -> pd.DataFrame:
        """
        Cluster texts based on embeddings.
        
        Args:
            texts: List of texts
            chunk_ids: IDs for each text
            embeddings: Numpy array of embeddings
            
        Returns:
            DataFrame with texts and cluster assignments
        """
        # If too few embeddings for meaningful clustering
        if len(embeddings) <= 5:
            # Assign all to one cluster
            cluster_labels = [np.array([0])] * len(embeddings)
            return pd.DataFrame({
                "text": texts,
                "chunk_id": chunk_ids,
                "cluster": cluster_labels
            })
        
        # Dimension reduction
        reduced_embeddings = self._global_cluster_embeddings(embeddings, 10)
        
        # Cluster with GMM
        n_clusters = min(max(2, len(embeddings) // 5), 20)  # Heuristic
        gm = GaussianMixture(n_components=n_clusters, random_state=self.random_seed)
        gm.fit(reduced_embeddings)
        
        # Get probabilistic cluster assignments
        probs = gm.predict_proba(reduced_embeddings)
        threshold = 0.1  # Membership threshold
        
        # Create multi-membership clusters
        cluster_labels = [np.where(prob > threshold)[0] for prob in probs]
        
        # Create DataFrame
        return pd.DataFrame({
            "text": texts,
            "chunk_id": chunk_ids,
            "cluster": cluster_labels
        })
    
    def _global_cluster_embeddings(
        self, 
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Perform dimensionality reduction on embeddings.
        
        Args:
            embeddings: Embedding vectors
            dim: Target dimension
            n_neighbors: UMAP neighbors parameter
            metric: Distance metric
            
        Returns:
            Reduced embeddings
        """
        if n_neighbors is None:
            n_neighbors = min(15, max(5, int((len(embeddings) - 1) ** 0.5)))
        
        return umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=dim,
            metric=metric,
            random_state=self.random_seed
        ).fit_transform(embeddings)
    
    def _format_texts_for_summary(self, texts: List[str], level: int) -> str:
        """
        Format texts for input to the summarization model.
        
        Args:
            texts: List of texts to format
            level: Current tree level
            
        Returns:
            Formatted text string
        """
        formatted_text = ""
        
        for i, text in enumerate(texts):
            section_header = f"--- SECTION {i+1} LEVEL {level-1} TEXT ---"
            formatted_text += f"{section_header}\n\n{text}\n\n"
        
        return formatted_text