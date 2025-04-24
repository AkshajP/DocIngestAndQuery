import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import umap
from sklearn.mixture import GaussianMixture

# From rag_system.py
from rag_system import DocumentProcessor, MilvusVectorStore, EmbeddingGenerator, EnhancedMilvusVectorStore
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    # handlers=[
    #     logging.FileHandler("raptor.log"),
    #     logging.StreamHandler()
    # ]
)
logger = logging.getLogger('raptor')

class Raptor:
    """
    RAPTOR (Recursive Abstractive Processing for Tree Organized Retrieval) implementation
    for hierarchical document retrieval with clustering and summarization.
    """
    
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2",
        ollama_embed_model: str = "llama3.2",
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        collection_name: str = "raptor_store",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_tree_levels: int = 3,
        random_seed: int = 224
    ):
        """Initialize the RAPTOR system with required components."""
        logger.info("Initializing RAPTOR system")
        self.random_seed = random_seed
        self.max_tree_levels = max_tree_levels
        
        # Document processor for loading and chunking
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Embedding generator
        self.embeddings = EmbeddingGenerator(
            model_name=ollama_embed_model,
            base_url=ollama_base_url
        )
        
        # Initialize the standard Milvus vector store
        standard_vector_store = MilvusVectorStore(
            host=milvus_host,
            port=milvus_port,
            collection_name=collection_name
        )
        
        # Wrap it with our enhanced version to handle large texts
        self.vector_store = EnhancedMilvusVectorStore(standard_vector_store)
        
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
        
        # Initialize storage
        self.tree = {}  # Hierarchical tree structure
        self.all_documents = {}  # All document chunks and summaries
        self.document_ids = set()  # Processed document IDs
        
        # Tree-specific mappings for retrieval
        self.node_to_layer = {}  # Maps node IDs to their layer in the tree
        self.layer_to_nodes = {}  # Maps layers to lists of nodes at that layer
        
        logger.info("RAPTOR initialization complete")
    
    def add_documents(self, file_paths: Union[str, List[str]]) -> int:
        """Process and add documents to the RAPTOR system."""
        # Convert single path to list if needed
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        logger.info(f"Adding {len(file_paths)} documents to RAPTOR")
        
        all_chunks = []
        processed_count = 0
        
        # Process each document
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            try:
                # Add document ID to tracking set
                self.document_ids.add(file_path)
                
                # Load and chunk document
                logger.info(f"Processing document: {file_path}")
                documents = self.document_processor.load_document(file_path)
                chunks = self.document_processor.split_documents(documents)
                
                # Add source file path to metadata
                for chunk in chunks:
                    if "source" not in chunk.metadata:
                        chunk.metadata["source"] = file_path
                
                all_chunks.extend(chunks)
                processed_count += len(chunks)
                
                logger.info(f"Document {file_path} processed into {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing document {file_path}: {str(e)}")
        
        if not all_chunks:
            logger.warning("No chunks were extracted from the documents")
            return 0
        
        # Build RAPTOR tree
        self._build_tree(all_chunks)
        
        return processed_count
    
    def delete_document(self, file_path: str) -> bool:
        """Delete a document from the RAPTOR system."""
        if file_path not in self.document_ids:
            logger.warning(f"Document not found in RAPTOR: {file_path}")
            return False
        
        try:
            # Remove document ID from tracking set
            self.document_ids.remove(file_path)
            logger.info(f"Deleting document: {file_path}")
            
            # Get all chunks from remaining documents
            all_chunks = []
            for doc_id in self.document_ids:
                documents = self.document_processor.load_document(doc_id)
                chunks = self.document_processor.split_documents(documents)
                all_chunks.extend(chunks)
            
            # Rebuild tree
            self._build_tree(all_chunks)
            
            logger.info(f"Document {file_path} deleted and tree rebuilt")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {file_path}: {str(e)}")
            return False
    
    def _build_tree(self, document_chunks: List[Document]):
        """Build the RAPTOR tree from document chunks, preserving bbox information."""
        logger.info(f"Building RAPTOR tree with {len(document_chunks)} chunks")
        
        # Reset tree and all_documents
        self.tree = {}
        self.all_documents = {}
        self.layer_to_nodes = {}
        self.node_to_layer = {}
        
        # Filter out chunks that appear to be just image references
        filtered_chunks = []
        for chunk in document_chunks:
            # Skip chunks that are primarily just image references
            content = chunk.page_content
            metadata_type = chunk.metadata.get("type", "")
            
            # Exclude pure image references with minimal content
            if metadata_type == "image" and len(content.strip().split()) < 15:
                logger.info(f"Skipping image reference with minimal content: {content[:50]}...")
                continue
                
            filtered_chunks.append(chunk)
        
        logger.info(f"Filtered {len(document_chunks) - len(filtered_chunks)} chunks that were primarily image references")
        document_chunks = filtered_chunks
        
        if not document_chunks:
            logger.warning("No valid chunks remain after filtering. Unable to build tree.")
            return
        
        # Extract text from filtered chunks for embedding
        texts = [chunk.page_content for chunk in document_chunks]
        
        # Store original document chunks as leaf nodes (layer 0)
        for i, chunk in enumerate(document_chunks):
            node_id = f"leaf_{i}"
            
            # Ensure original_boxes is preserved in metadata
            metadata_copy = chunk.metadata.copy() if chunk.metadata else {}
            
            self.all_documents[node_id] = {
                "text": chunk.page_content,
                "metadata": metadata_copy,  # Contains original_boxes if present
                "level": 0,
                "is_original": True,
                "children": set()  # No children for leaf nodes
            }
        
        # Create layer 0 mapping
        layer_0_nodes = [node_id for node_id, info in self.all_documents.items() if info["level"] == 0]
        self.layer_to_nodes[0] = layer_0_nodes
        
        # Update node to layer mapping for layer 0
        for node_id in layer_0_nodes:
            self.node_to_layer[node_id] = 0
        
        # Start recursive tree building
        tree_results = self._recursive_embed_cluster_summarize(texts, level=1)
        
        # Store tree structure
        self.tree = tree_results
        
        # Combine all documents and summaries for vector storage
        all_texts = []
        all_metadatas = []
        
        # Add original document chunks
        for doc_id, doc_info in self.all_documents.items():
            if doc_info["is_original"]:
                all_texts.append(doc_info["text"])
                
                # Create metadata including original_boxes if present
                metadata = {
                    "id": doc_id,
                    "level": doc_info["level"],
                    "source": doc_info["metadata"].get("source", "unknown"),
                    "is_original": True
                }
                
                # Copy original_boxes if present
                if "original_boxes" in doc_info["metadata"]:
                    metadata["original_boxes"] = doc_info["metadata"]["original_boxes"]
                
                all_metadatas.append(metadata)
        
        # Add summaries from all levels
        for level, (clusters_df, summaries_df) in self.tree.items():
            # Create layer mapping for this level
            layer_nodes = []
            
            for i, row in summaries_df.iterrows():
                summary_id = f"summary_l{level}_c{row['cluster']}"
                summary_text = row['summaries']
                
                # Get children (nodes from previous level that are part of this cluster)
                # This makes node relationships explicit for hierarchical traversal
                child_indices = clusters_df[clusters_df['cluster'].apply(lambda x: row['cluster'] in x)].index.tolist()
                child_nodes = [f"leaf_{idx}" if level == 1 else f"summary_l{level-1}_c{idx}" for idx in child_indices]
                
                # Collect original_boxes from all children for this summary node
                all_original_boxes = []
                for child_node in child_nodes:
                    if child_node in self.all_documents:
                        child_metadata = self.all_documents[child_node]["metadata"]
                        if "original_boxes" in child_metadata:
                            all_original_boxes.extend(child_metadata["original_boxes"])
                
                # Add to all_documents with original_boxes
                node_metadata = {"level": level, "cluster": row["cluster"]}
                if all_original_boxes:
                    node_metadata["original_boxes"] = all_original_boxes
                
                self.all_documents[summary_id] = {
                    "text": summary_text,
                    "metadata": node_metadata,
                    "level": level,
                    "is_original": False,
                    "children": set(child_nodes)
                }
                
                # Add to layer nodes list
                layer_nodes.append(summary_id)
                
                # Update node_to_layer mapping
                self.node_to_layer[summary_id] = level
                
                # Add to lists for vector store
                all_texts.append(summary_text)
                
                # Prepare metadata for vector store
                summary_metadata = {
                    "id": summary_id,
                    "level": level,
                    "cluster": row["cluster"],
                    "is_original": False
                }
                
                # Include original_boxes if available
                if all_original_boxes:
                    summary_metadata["original_boxes"] = all_original_boxes
                    
                all_metadatas.append(summary_metadata)
            
            # Update layer_to_nodes mapping
            self.layer_to_nodes[level] = layer_nodes
        
        # Generate embeddings for all texts
        logger.info(f"Generating embeddings for {len(all_texts)} documents and summaries")
        embeddings = self._embed_documents(all_texts)
        
        # Store in vector database
        logger.info("Storing documents and summaries in vector database")
        vector_documents = []
        
        for i, (text, metadata) in enumerate(zip(all_texts, all_metadatas)):
            vector_documents.append(Document(page_content=text, metadata=metadata))
        
        # Our enhanced vector store handles text length limits automatically
        self.vector_store.add_documents(vector_documents, embeddings)
        
        logger.info("RAPTOR tree construction complete")
    
    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        return self.embeddings.generate_embeddings(texts)
    
    def _global_cluster_embeddings(
        self, 
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine"
    ) -> np.ndarray:
        """Perform global dimensionality reduction on embeddings using UMAP."""
        if n_neighbors is None:
            n_neighbors = int((len(embeddings) - 1) ** 0.5)
        
        return umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=dim,
            metric=metric,
            random_state=self.random_seed
        ).fit_transform(embeddings)
    
    def _local_cluster_embeddings(
        self,
        embeddings: np.ndarray,
        dim: int,
        num_neighbors: int = 10,
        metric: str = "cosine"
    ) -> np.ndarray:
        """Perform local dimensionality reduction on embeddings using UMAP."""
        return umap.UMAP(
            n_neighbors=num_neighbors,
            n_components=dim,
            metric=metric,
            random_state=self.random_seed
        ).fit_transform(embeddings)
    
    def _get_optimal_clusters(
        self,
        embeddings: np.ndarray,
        max_clusters: int = 50
    ) -> int:
        """Determine optimal number of clusters using BIC."""
        max_clusters = min(max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=self.random_seed)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
            
        return n_clusters[np.argmin(bics)]
    
    def _gmm_cluster(
        self,
        embeddings: np.ndarray,
        threshold: float
    ) -> Tuple[List[np.ndarray], int]:
        """Cluster embeddings using Gaussian Mixture Model."""
        n_clusters = self._get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=self.random_seed)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        
        return labels, n_clusters
    
    def _perform_clustering(
        self,
        embeddings: np.ndarray,
        dim: int,
        threshold: float
    ) -> List[np.ndarray]:
        """Perform hierarchical clustering on embeddings."""
        if len(embeddings) <= dim + 1:
            # Avoid clustering with insufficient data
            return [np.array([0]) for _ in range(len(embeddings))]
        
        # Global dimensionality reduction
        reduced_embeddings_global = self._global_cluster_embeddings(embeddings, dim)
        
        # Global clustering
        global_clusters, n_global_clusters = self._gmm_cluster(reduced_embeddings_global, threshold)
        
        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0
        
        # Iterate through each global cluster for local clustering
        for i in range(n_global_clusters):
            # Extract embeddings belonging to current global cluster
            global_cluster_mask = np.array([i in gc for gc in global_clusters])
            if not np.any(global_cluster_mask):
                continue
                
            global_cluster_embeddings = embeddings[global_cluster_mask]
            
            if len(global_cluster_embeddings) <= dim + 1:
                # Handle small clusters with direct assignment
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings]
                n_local_clusters = 1
            else:
                # Local dimensionality reduction and clustering
                reduced_embeddings_local = self._local_cluster_embeddings(global_cluster_embeddings, dim)
                local_clusters, n_local_clusters = self._gmm_cluster(reduced_embeddings_local, threshold)
            
            # Find indices of original embeddings
            indices = np.where(global_cluster_mask)[0]
            
            # Assign local cluster IDs
            for j, idx in enumerate(indices):
                if j < len(local_clusters):
                    for cluster in local_clusters[j]:
                        all_local_clusters[idx] = np.append(all_local_clusters[idx], cluster + total_clusters)
            
            total_clusters += n_local_clusters
        
        return all_local_clusters
    
    def _embed_cluster_texts(self, texts: List[str]) -> pd.DataFrame:
        """Embed and cluster texts."""
        # Generate embeddings
        text_embeddings = self._embed_documents(texts)
        text_embeddings_np = np.array(text_embeddings)
        
        # Perform clustering
        cluster_labels = self._perform_clustering(text_embeddings_np, 10, 0.1)
        
        # Create DataFrame
        df = pd.DataFrame()
        df["text"] = texts
        df["embd"] = list(text_embeddings_np)
        df["cluster"] = cluster_labels
        
        return df
    
    def _format_texts(self, df: pd.DataFrame) -> str:
        """
        Format texts from a DataFrame into a single string with rich context.
        Includes metadata and separates different content types for better summarization.
        """
        formatted_texts = []
        
        for index, row in df.iterrows():
            text = row["text"]
            
            # Check if this is a node from a previous level that we can retrieve
            if isinstance(text, str) and text.startswith("summary_l"):
                node_id = text
                if node_id in self.all_documents:
                    # Get the full content including metadata
                    node_info = self.all_documents[node_id]
                    
                    # Add header with level information
                    level = node_info.get("level", "unknown")
                    formatted_texts.append(f"--- SUMMARY FROM LEVEL {level} ---")
                    
                    # Add the actual text
                    formatted_texts.append(node_info.get("text", ""))
                else:
                    # Just use the text as-is if we can't find it
                    formatted_texts.append(text)
            else:
                # Handle original text with metadata if available
                node_metadata = {}
                for i, node_id in enumerate(self.layer_to_nodes.get(0, [])):
                    if self.all_documents.get(node_id, {}).get("text") == text:
                        node_metadata = self.all_documents.get(node_id, {}).get("metadata", {})
                        break
                
                # Add type-specific formatting
                content_type = node_metadata.get("type", "text")
                source = node_metadata.get("source", "unknown")
                page = node_metadata.get("page_idx", "")
                
                # Add header with content type and source
                header = f"--- {content_type.upper()} "
                if page:
                    header += f"(PAGE {page}) "
                header += f"FROM {source} ---"
                formatted_texts.append(header)
                
                # Add the content
                formatted_texts.append(text)
        
        # Join with clear separators and spacing
        return "\n\n".join(formatted_texts)
    
    def _embed_cluster_summarize_texts(
        self,
        texts: List[str],
        level: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Embed, cluster, and summarize texts, ensuring at least one summary per level."""
        logger.info(f"Processing level {level} with {len(texts)} texts")
        
        # Handle edge case with very few texts
        if len(texts) <= 1 and level > 1:
            # For levels above 1, if we only have one text, still create a summary
            # Create a simple "cluster" (no actual clustering needed)
            dummy_embedding = self.embeddings.embed_query(texts[0])
            df_clusters = pd.DataFrame({
                "text": texts,
                "embd": [dummy_embedding],
                "cluster": [[0]]  # Single cluster
            })
            
            # Generate a summary
            formatted_txt = f"--- SUMMARY FROM LEVEL {level-1} ---\n\n{texts[0]}"
            try:
                summary_chain = self.summary_prompt | self.llm | StrOutputParser()
                summary = summary_chain.invoke({"context": formatted_txt})
                logger.debug(f"Generated summary for single text at level {level}")
            except Exception as e:
                logger.error(f"Error generating summary for single text: {str(e)}")
                summary = f"Summary of: {texts[0][:100]}..."  # Fallback summary
            
            # Create summary DataFrame
            df_summary = pd.DataFrame({
                "summaries": [summary],
                "level": [level],
                "cluster": [0]
            })
            
            return df_clusters, df_summary
        
        # Original clustering and summarization for multiple texts
        df_clusters = self._embed_cluster_texts(texts)
        
        # Prepare expanded DataFrame for easier manipulation
        expanded_list = []
        
        # Expand DataFrame entries to document-cluster pairings
        for index, row in df_clusters.iterrows():
            for cluster in row["cluster"]:
                expanded_list.append({
                    "text": row["text"],
                    "embd": row["embd"],
                    "cluster": cluster
                })
        
        # Create expanded DataFrame
        expanded_df = pd.DataFrame(expanded_list)
        
        # Initialize df_summary as an empty DataFrame in case we don't create any summaries
        df_summary = pd.DataFrame(columns=["summaries", "level", "cluster"])
        
        # Get unique cluster identifiers
        all_clusters = expanded_df["cluster"].unique() if not expanded_df.empty else []
        
        logger.info(f"Generated {len(all_clusters)} clusters at level {level}")
        
        # Format text within each cluster and generate summaries
        summaries = []
        cluster_ids = []
        for i in all_clusters:
            df_cluster = expanded_df[expanded_df["cluster"] == i]
            formatted_txt = self._format_texts(df_cluster)
            
            try:
                summary_chain = self.summary_prompt | self.llm | StrOutputParser()
                summary = summary_chain.invoke({"context": formatted_txt})
                summaries.append(summary)
                cluster_ids.append(i)
                logger.debug(f"Generated summary for cluster {i} at level {level}")
            except Exception as e:
                logger.error(f"Error generating summary for cluster {i}: {str(e)}")
                # Still add the cluster but with an error message as summary
                summaries.append(f"Error generating summary: {str(e)}")
                cluster_ids.append(i)
        
        # Create DataFrame for summaries if we have any
        if summaries:
            df_summary = pd.DataFrame({
                "summaries": summaries,
                "level": [level] * len(summaries),
                "cluster": cluster_ids
            })
        
        # Final check: ensure we created at least one summary
        if df_summary.empty and texts:
            # Create at least one summary using all texts
            all_text = "\n\n".join(texts)
            formatted_txt = f"--- COMBINED TEXT FOR LEVEL {level} ---\n\n{all_text}"
            
            try:
                summary_chain = self.summary_prompt | self.llm | StrOutputParser()
                summary = summary_chain.invoke({"context": formatted_txt})
            except Exception as e:
                logger.error(f"Error generating fallback summary: {str(e)}")
                summary = f"Combined summary from level {level}"
                
            # Create a summary DataFrame with at least one entry
            df_summary = pd.DataFrame({
                "summaries": [summary],
                "level": [level],
                "cluster": [0]
            })
        
        return df_clusters, df_summary
    
    def _recursive_embed_cluster_summarize(
        self,
        texts: List[str],
        level: int = 1
    ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Recursively embed, cluster, and summarize texts up to max levels."""
        results = {}
        
        # Process current level
        df_clusters, df_summary = self._embed_cluster_summarize_texts(texts, level)
        
        # Store results
        results[level] = (df_clusters, df_summary)
        
        # Modified condition: Continue building tree as long as we haven't reached max_level
        # and we have at least one text to summarize
        if level < self.max_tree_levels and len(texts) > 0:
            # Use summaries as input for next level
            new_texts = df_summary["summaries"].tolist()
            next_level_results = self._recursive_embed_cluster_summarize(new_texts, level + 1)
            
            # Merge results
            results.update(next_level_results)
        
        return results
    
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        # Simple approximation for token count (you may want to use a real tokenizer)
        return len(text.split())
    
    def _get_node_text(self, node_id: str) -> str:
        """Get the text content of a node."""
        if node_id in self.all_documents:
            return self.all_documents[node_id]["text"]
        return ""
    
    def _get_node_children(self, node_id: str) -> Set[str]:
        """Get the children of a node."""
        if node_id in self.all_documents:
            return self.all_documents[node_id]["children"]
        return set()
    
    def _get_nodes_at_layer(self, layer: int) -> List[str]:
        """Get all node IDs at a specific layer."""
        return self.layer_to_nodes.get(layer, [])
    
    def _calculate_similarity(self, query_embedding: List[float], node_embedding: List[float]) -> float:
        """Calculate cosine similarity between query and node embeddings."""
        # Compute cosine similarity (dot product of normalized vectors)
        dot_product = sum(a * b for a, b in zip(query_embedding, node_embedding))
        magnitude1 = sum(a * a for a in query_embedding) ** 0.5
        magnitude2 = sum(b * b for b in node_embedding) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)
    
    def _retrieve_information_collapse_tree(
        self, 
        query: str, 
        top_k: int, 
        max_tokens: int
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Retrieve information using the collapsed tree approach (treats tree as flat).
        Preserves bbox information in results.
        
        Args:
            query: The query string
            top_k: Maximum number of nodes to retrieve
            max_tokens: Maximum tokens to include in the context
            
        Returns:
            Tuple of (selected_nodes_info, context_text)
        """
        logger.info(f"Using collapsed tree retrieval with max_tokens={max_tokens}")
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Retrieve similar documents from vector store
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # Format results and track token count
        selected_nodes = []
        context_parts = []
        total_tokens = 0
        
        for result in results:
            node_text = result["content"]
            node_tokens = self._count_tokens(node_text)
            
            # Check if adding this node would exceed the token budget
            if total_tokens + node_tokens > max_tokens:
                # If we've already got at least one node, stop here
                if selected_nodes:
                    break
                # Otherwise, truncate the text to fit (better to have some context than none)
                else:
                    # Simple truncation - in practice you'd want a more sophisticated approach
                    node_text = " ".join(node_text.split()[:max_tokens])
                    node_tokens = max_tokens
            
            # Add node to results - now preserving original_boxes
            metadata = result["metadata"].copy()
            
            node_info = {
                "content": node_text,
                "metadata": metadata,  # This now includes original_boxes if present
                "score": result["score"]
            }
            
            selected_nodes.append(node_info)
            context_parts.append(node_text)
            total_tokens += node_tokens
        
        # Combine context parts
        context = "\n\n".join(context_parts)
        
        logger.info(f"Retrieved {len(selected_nodes)} nodes with {total_tokens} tokens using collapsed tree")
        return selected_nodes, context

    def _retrieve_information_hierarchical(
        self, 
        query: str, 
        start_layer: int,
        num_layers: int,
        top_k: int, 
        max_tokens: int
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Retrieve information using hierarchical tree traversal with strong preference for summary nodes.
        """
        logger.info(f"Using dynamic hierarchical traversal from layer {start_layer}")
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Calculate the range of layers to consider
        end_layer = max(0, start_layer - (num_layers - 1))
        layers_to_consider = range(start_layer, end_layer - 1, -1)
        
        logger.info(f"Considering layers from {start_layer} down to {end_layer}")
        
        # Create a pool of candidate nodes from all relevant layers
        all_candidate_nodes = []
        for layer in layers_to_consider:
            if layer in self.layer_to_nodes:
                layer_nodes = self.layer_to_nodes[layer]
                all_candidate_nodes.extend([(node_id, layer) for node_id in layer_nodes])
        
        # Calculate similarity for all candidate nodes
        node_similarities = []
        for node_id, layer in all_candidate_nodes:
            # Get node embedding and text
            node_text = self._get_node_text(node_id)
            node_embedding = self.embeddings.embed_query(node_text)
            similarity = self._calculate_similarity(query_embedding, node_embedding)
            node_tokens = self._count_tokens(node_text)
            
            # Get node metadata
            node_metadata = self.all_documents.get(node_id, {}).get("metadata", {})
            
            # Determine if this is a summary node or leaf node
            is_summary = layer > 0  # Layer 0 contains leaf nodes
            
            # Store all the information we need for sorting and selection
            node_info = {
                "id": node_id,
                "layer": layer,
                "similarity": similarity,
                "content": node_text,
                "tokens": node_tokens,
                "metadata": node_metadata,
                "is_summary": is_summary,
                # MODIFIED: Apply much stronger bias for summary nodes
                # Use boosting for layers and a flat boost for summaries
                "adjusted_score": similarity + (2.0 * layer) * (3.0 if is_summary else 1.0)
            }
            node_similarities.append(node_info)
        
        # Sort all nodes based on adjusted score
        node_similarities.sort(key=lambda x: x["adjusted_score"], reverse=True)
        
        # MODIFIED: First get summaries, then leaf nodes if needed
        summary_nodes = [node for node in node_similarities if node["is_summary"]]
        leaf_nodes = [node for node in node_similarities if not node["is_summary"]]
        
        # Prioritize summary nodes, but include leaf nodes if we need more
        prioritized_nodes = summary_nodes + leaf_nodes
        
        # Now select nodes, respecting the token budget
        selected_nodes = []
        context_parts = []
        total_tokens = 0
        
        # Take top summary nodes first
        # Process nodes in order of priority
        top_candidates = prioritized_nodes[:top_k * 3]  # Get a wider pool than top_k
        
        for node in top_candidates:
            # Check if adding this node would exceed the token budget
            if total_tokens + node["tokens"] > max_tokens:
                # If we've already got some nodes, skip this one
                if selected_nodes:
                    continue
                # Otherwise, truncate the first node to fit
                else:
                    words = node["content"].split()
                    truncated_text = " ".join(words[:max_tokens])
                    node["content"] = truncated_text
                    node["tokens"] = max_tokens
            
            # Add node to results
            node_info = {
                "id": node["id"],
                "content": node["content"],
                "layer": node["layer"],
                "score": node["similarity"],
                "is_summary": node["is_summary"],  # Added for clarity in results
                "metadata": node["metadata"]
            }
            selected_nodes.append(node_info)
            context_parts.append(node["content"])
            total_tokens += node["tokens"]
            
            # If we have enough nodes, stop
            if len(selected_nodes) >= top_k:
                break
        
        # Fallback if no nodes selected
        if not selected_nodes and node_similarities:
            # Prefer a summary node for fallback if available
            fallback_node = next((node for node in node_similarities if node["is_summary"]), node_similarities[0])
            
            # Truncate if needed
            if fallback_node["tokens"] > max_tokens:
                words = fallback_node["content"].split()
                truncated_text = " ".join(words[:max_tokens])
                node_tokens = max_tokens
            else:
                truncated_text = fallback_node["content"]
                node_tokens = fallback_node["tokens"]
            
            node_info = {
                "id": fallback_node["id"],
                "content": truncated_text,
                "layer": fallback_node["layer"],
                "score": fallback_node["similarity"],
                "is_summary": fallback_node.get("is_summary", False),
                "metadata": fallback_node["metadata"]
            }
            selected_nodes.append(node_info)
            context_parts.append(truncated_text)
            total_tokens = node_tokens
        
        # Combine text from all selected nodes
        context = "\n\n".join(context_parts)
        
        # Log retrieval statistics
        summary_count = len([node for node in selected_nodes if node.get("is_summary", False)])
        leaf_count = len(selected_nodes) - summary_count
        
        logger.info(f"Retrieved {len(selected_nodes)} nodes ({summary_count} summaries, {leaf_count} leaves)")
        logger.info(f"Total tokens: ~{total_tokens}")
        
        return selected_nodes, context
    
    def retrieve_chunks(
        self,
        query: str,
        top_k: int = 5,
        max_tokens: int = 3000,
        collapse_tree: bool = False,  # Changed default to False to prefer hierarchical
        start_layer: Optional[int] = None,
        num_layers: Optional[int] = None,
        summary_preference: float = 3.0  # New parameter to control summary preference
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Retrieve relevant chunks for a query with strong preference for summary nodes.
        
        Args:
            query: The query string
            top_k: Maximum number of chunks to retrieve
            max_tokens: Maximum number of tokens in the retrieved context
            collapse_tree: Whether to use collapsed tree retrieval (False for hierarchical)
            start_layer: Starting layer for hierarchical traversal (highest by default)
            num_layers: Number of layers to traverse in hierarchical mode
            summary_preference: Multiplier for summary node scores (higher = stronger preference)
            
        Returns:
            Tuple of (retrieved_chunks, context_text)
        """
        logger.info(f"Retrieving chunks for query: {query} with summary_preference={summary_preference}")
        
        # Set default values for hierarchical traversal
        if not collapse_tree:
            if start_layer is None:
                # Start from the highest available layer
                start_layer = max(self.layer_to_nodes.keys()) if self.layer_to_nodes else 0
                    
            if num_layers is None:
                # Default to traversing all available layers
                num_layers = start_layer + 1
        
        # Validate parameters
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            logger.warning(f"Invalid max_tokens: {max_tokens}, using default 3500")
            max_tokens = 3000
                
        if not isinstance(top_k, int) or top_k <= 0:
            logger.warning(f"Invalid top_k: {top_k}, using default 5")
            top_k = 5
        
        # Select retrieval method based on collapse_tree parameter
        if collapse_tree:
            logger.info(f"Using collapsed tree retrieval with top_k={top_k}, max_tokens={max_tokens}")
            selected_nodes, context = self._retrieve_information_collapse_tree(
                query=query,
                top_k=top_k,
                max_tokens=max_tokens
            )
        else:
            logger.info(f"Using hierarchical traversal from layer {start_layer} with {num_layers} layers")
            # Set the summary preference as a thread-local variable or pass to the method
            # For this example, we'll modify the method call
            selected_nodes, context = self._retrieve_information_hierarchical(
                query=query,
                start_layer=start_layer,
                num_layers=num_layers,
                top_k=top_k,
                max_tokens=max_tokens
            )
            
        return selected_nodes, context
    
    def cleanup(self):
        """Release resources."""
        logger.info("Cleaning up RAPTOR resources")
        self.vector_store.release()


if __name__ == "__main__":
    
    # Initialize RAPTOR
    logger.info("Initializing RAPTOR with enhanced Milvus support")
    raptor = Raptor(
        collection_name="raptor_test",  # Use a unique collection name for testing
        max_tree_levels=3
    )
    
    # Find PDF documents in the current directory
    doc_files = ['abc.md'] # [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if not doc_files:
        logger.warning("No PDF documents found in the current directory.")
        logger.info("Using example documents provided by command line arguments...")
        # Get files from command line arguments
        import sys
        if len(sys.argv) > 1:
            doc_files = sys.argv[1:]
        else:
            logger.error("No documents provided! Please specify document paths as arguments.")
            os.exit(1)
    
    # Process documents
    logger.info(f"Processing {len(doc_files)} documents")
    chunks_processed = raptor.add_documents(doc_files)
    logger.info(f"Processed {chunks_processed} chunks across all documents")
    
    # Define test queries
    predefined_queries = [
        {"query": "explain exhibit 17", "method": "collapse"},
        {"query": "explain exhibit 17", "method": "hierarchical"},
        {"query": "Summarize the key findings", "method": "collapse"}
    ]
    
    # Test both retrieval methods
    logger.info("Processing predefined queries...")
    
    for query_info in predefined_queries:
        query = query_info["query"]
        method = query_info["method"]
        
        # Set collapse_tree based on the predefined method
        collapse_tree = method == "collapse"
        
        logger.info(f"\nQuery: {query}")
        logger.info(f"Using method: {method}")
        
        # Retrieve chunks using the predefined method
        chunks, context = raptor.retrieve_chunks(
            query=query,
            top_k=5,
            max_tokens=2000,
            collapse_tree=collapse_tree,
            # For hierarchical queries, we can optionally set these parameters
            start_layer=None if method == "collapse" else 2,
            num_layers=None if method == "collapse" else 3
        )
        
        # Print results
        logger.info(f"Retrieved {len(chunks)} chunks using {method} method")
        
        # Print a sample of the retrieved content
        if chunks:
            logger.info(f"First chunk (score: {chunks[0].get('score', 'N/A')}):")
            content = chunks[0].get('content', '')
            logger.info(f"Content (first 150 chars): {content[:150]}...")
    
    # Clean up
    logger.info("Test complete. Cleaning up resources...")
    raptor.cleanup()