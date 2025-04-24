import os
import json
import time
import pickle
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import shutil
import tempfile
import re
from tabulate import tabulate  # For nicer table formatting

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('document_query')

# Import Ollama components
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Add this class after the existing imports in documentQueryDecoupled.py
class ChatHistory:
    """Manages conversation history with progressive summarization based on relevance"""
    
    def __init__(
        self, 
        max_history_length: int = 5,
        llm = None,
        embeddings = None
    ):
        """
        Initialize chat history manager with summarization capability.
        
        Args:
            max_history_length: Maximum number of individual interactions before summarization
            llm: LLM instance for summarization (will be set later if None)
            embeddings: Embeddings instance for relevance calculation (will be set later if None)
        """
        self.history = []
        self.summarized_history = []
        self.max_history_length = max_history_length
        self.llm = llm
        self.embeddings = embeddings
    
    def set_llm(self, llm):
        """Set the LLM instance for summarization"""
        self.llm = llm
    
    def set_embeddings(self, embeddings):
        """Set the embeddings instance for relevance calculation"""
        self.embeddings = embeddings
    
    def add_interaction(self, question: str, answer: str):
        """
        Add a Q&A interaction to the history and manage summarization.
        
        Args:
            question: User question
            answer: System answer
        """
        # Add new interaction
        self.history.append({"question": question, "answer": answer})
        
        # Check if we need to summarize
        if len(self.history) > self.max_history_length:
            self._progressive_summarize(question)
    
    def _progressive_summarize(self, current_query: str):
        """
        Progressively summarize history based on relevance to current query.
        
        Args:
            current_query: The current question for relevance calculation
        """
        if not self.llm:
            # If LLM not available, just trim without summarization
            self._simple_trim()
            return
        
        # Calculate number of interactions to summarize (half of max, rounded up)
        num_to_summarize = (self.max_history_length + 1) // 2
        
        # Extract interactions to summarize
        to_summarize = self.history[:num_to_summarize]
        remaining = self.history[num_to_summarize:]
        
        # Evaluate relevance if embeddings are available
        relevance_scores = None
        if self.embeddings and current_query:
            relevance_scores = self._calculate_relevance(to_summarize, current_query)
        
        # Prepare the content for summarization
        summary_content = ""
        for i, interaction in enumerate(to_summarize):
            importance = "high" if relevance_scores and relevance_scores[i] > 0.7 else "normal"
            summary_content += f"USER: {interaction['question']}\n"
            summary_content += f"ASSISTANT: {interaction['answer']}\n"
            summary_content += f"IMPORTANCE: {importance}\n\n"
        
        # Create the summarization prompt
        prompt = f"""
        Summarize the following conversation history. Focus on key information, questions, and findings.
        Preserve details that might be relevant to future questions.
        Pay special attention to information marked as high importance.
        
        CONVERSATION HISTORY:
        {summary_content}
        
        SUMMARY:
        """
        
        # Generate summary
        try:
            from langchain_core.output_parsers import StrOutputParser
            summary = (self.llm | StrOutputParser()).invoke(prompt)
            
            # Add summary to summarized history
            self.summarized_history.append({
                "type": "summary",
                "content": summary,
                "original_count": len(to_summarize)
            })
            
            # Update history with remaining items
            self.history = remaining
            
        except Exception as e:
            # Fallback to simple trimming if summarization fails
            print(f"Error during history summarization: {str(e)}")
            self._simple_trim()
    
    def _calculate_relevance(self, interactions, current_query):
        """
        Calculate relevance scores between past interactions and current query.
        
        Args:
            interactions: List of past interactions
            current_query: Current user query
            
        Returns:
            List of relevance scores (0-1)
        """
        try:
            # Generate embedding for current query
            query_embedding = self.embeddings.embed_query(current_query)
            
            # Calculate relevance for each interaction
            scores = []
            for interaction in interactions:
                # Combine question and answer for context
                text = f"{interaction['question']} {interaction['answer']}"
                text_embedding = self.embeddings.embed_query(text)
                
                # Calculate cosine similarity
                import numpy as np
                dot_product = np.dot(query_embedding, text_embedding)
                norm1 = np.linalg.norm(query_embedding)
                norm2 = np.linalg.norm(text_embedding)
                
                if norm1 == 0 or norm2 == 0:
                    scores.append(0.0)
                else:
                    similarity = dot_product / (norm1 * norm2)
                    scores.append(float(similarity))
            
            return scores
            
        except Exception as e:
            print(f"Error calculating relevance: {str(e)}")
            return None
    
    def _simple_trim(self):
        """Simple trimming method as fallback"""
        # Just keep the most recent max_history_length interactions
        if len(self.history) > self.max_history_length:
            excess = len(self.history) - self.max_history_length
            self.history = self.history[excess:]
        
    def get_formatted_history(self) -> str:
        if not self.history and not self.summarized_history:
            return ""
        
        formatted = "Previous Conversation:\n"
        
        for summary in self.summarized_history:
            formatted += f"[Summary of previous interactions: {summary['content']}]\n\n"
        
        for interaction in self.history:
            formatted += f"Human: {interaction['question']}\n"
            formatted += f"Assistant: {interaction['answer']}\n\n"
        
        formatted += "Please refer to our previous conversation as needed when answering.\n"
        
        return formatted
    
    def clear(self):
        """Clear the chat history"""
        self.history = []
        self.summarized_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the current chat history.
        
        Returns:
            List of interaction dictionaries and summary dictionaries
        """
        full_history = []
        
        # Add summarized history with type marker
        for summary in self.summarized_history:
            full_history.append(summary)
        
        # Add individual interactions
        for interaction in self.history:
            full_history.append({**interaction, "type": "interaction"})
        
        return full_history
    
class DiskBasedDocumentQuerier:
    """
    Query interface that loads documents directly from disk storage.
    Efficient document loading and memory management.
    """
    
    def __init__(
        self,
        storage_dir: str = "document_store",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "deepseek-r1:14b",
        ollama_embed_model: str = "llama3.2",
        cache_dir: Optional[str] = None,
        max_history_length: int = 5
    ):
        """
        Initialize the document query interface.
        
        Args:
            storage_dir: Base directory for stored documents
            ollama_base_url: URL for Ollama API
            ollama_model: Model for LLM operations
            ollama_embed_model: Model for embeddings
            cache_dir: Directory for caching loaded components (None = use temp dir)
            max_history_length: Maximum number of interactions before summarization
        """
        self.storage_dir = storage_dir
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.ollama_embed_model = ollama_embed_model
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
        else:
            # Use system temp directory
            self.cache_dir = tempfile.mkdtemp(prefix="doc_query_cache_")
        
        # Check if storage directory exists
        if not os.path.exists(storage_dir):
            logger.warning(f"Storage directory {storage_dir} not found")
            os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize registry path
        self.registry_path = os.path.join(storage_dir, "document_registry.json")
        if not os.path.exists(self.registry_path):
            logger.warning(f"Document registry not found at {self.registry_path}")
            with open(self.registry_path, "w") as f:
                json.dump({"documents": {}, "last_updated": datetime.now().isoformat()}, f, indent=2)
        
        # Initialize embedding model
        self.embeddings = OllamaEmbeddings(
            model=ollama_embed_model,
            base_url=ollama_base_url
        )
        
        # Initialize LLM
        self.llm = OllamaLLM(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.1
        )
        
        # Create RAG prompt template with chat history
        self.prompt_template = PromptTemplate(
            template="""
                <system>
                You are a document analysis assistant. Answer the user's question based on the information from the documents I've indexed for you and the previous conversation.
                Only use information from the documents shown below and refer to previous conversations if needed. If the answer cannot be formed satisfactorily, acknowledge that you don't have enough information.
                Never claim the user wrote or said anything that appears in the documents.
                </system>

                <previous_conversation>
                {chat_history}
                </previous_conversation>

                <indexed_documents>
                {context}
                </indexed_documents>

                <user_query>
                {question}
                </user_query>

                <assistant>
                """,
                input_variables=["context", "question", "chat_history"]
            )      
        
        # Initialize chat history with LLM and embeddings for summarization
        self.chat_history = ChatHistory(max_history_length=max_history_length)
        self.chat_history.set_llm(self.llm)
        self.chat_history.set_embeddings(self.embeddings)
        
        # Cache for loaded documents
        self.document_cache = {}
        
        # Track loaded documents and components
        self.loaded_documents = set()
        self.document_metadata = {}
        self.chunk_cache = {}
        
        logger.info(f"Document query interface initialized with chat history and cache dir: {self.cache_dir}")    
    def refresh_registry(self) -> bool:
        """
        Refresh the document registry from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, "r") as f:
                    registry = json.load(f)
                
                # Update local metadata
                for doc_id, doc_info in registry.get("documents", {}).items():
                    self.document_metadata[doc_id] = doc_info
                
                return True
            else:
                logger.warning(f"Registry file not found: {self.registry_path}")
                return False
        except Exception as e:
            logger.error(f"Error refreshing registry: {str(e)}")
            return False
    
    def list_available_documents(self, refresh: bool = True) -> List[Dict[str, Any]]:
        """
        List all available documents from disk.
        
        Args:
            refresh: Whether to refresh registry from disk
            
        Returns:
            List of document information dictionaries
        """
        if refresh:
            self.refresh_registry()
        
        try:
            # Read the registry file
            with open(self.registry_path, "r") as f:
                registry = json.load(f)
            
            # Get document list and sort by date (newest first)
            documents = list(registry.get("documents", {}).values())
            documents.sort(key=lambda x: x.get("processing_date", ""), reverse=True)
            
            # Add loading status
            for doc in documents:
                doc_id = doc.get("document_id")
                doc["is_loaded"] = doc_id in self.loaded_documents
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    def load_document(self, document_id: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load a document directly from disk storage.
        
        Args:
            document_id: ID of document to load
            force_reload: Whether to force reload if already loaded
            
        Returns:
            Status dictionary
        """
        # Check if already loaded
        if document_id in self.loaded_documents and not force_reload:
            return {
                "status": "success",
                "message": f"Document {document_id} is already loaded",
                "document_id": document_id,
                "load_time": 0
            }
        
        # Path to document directory
        doc_dir = os.path.join(self.storage_dir, document_id)
        
        if not os.path.exists(doc_dir):
            return {
                "status": "error",
                "message": f"Document directory not found: {doc_dir}",
                "document_id": document_id
            }
        
        start_time = time.time()
        
        try:
            # Load document metadata
            metadata_path = os.path.join(doc_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.document_metadata[document_id] = metadata
            else:
                logger.warning(f"Metadata file not found for document {document_id}")
                metadata = {
                    "document_id": document_id,
                    "status": "unknown",
                    "chunks_count": 0
                }
                self.document_metadata[document_id] = metadata
            
            # Load minimal required components
            # We'll load other components on-demand to save memory
            
            # Check if tree structure exists
            tree_path = os.path.join(doc_dir, "tree_structure.json")
            if os.path.exists(tree_path):
                # Instead of loading the full tree, just check it exists
                tree_exists = True
            else:
                tree_exists = False
                logger.warning(f"Tree structure not found for document {document_id}")
            
            # Add to loaded documents set
            self.loaded_documents.add(document_id)
            
            # Cache the document directory path
            self.document_cache[document_id] = {
                "doc_dir": doc_dir,
                "metadata": metadata,
                "tree_exists": tree_exists
            }
            
            load_time = time.time() - start_time
            
            return {
                "status": "success",
                "message": f"Document {document_id} loaded successfully",
                "document_id": document_id,
                "chunks_count": metadata.get("chunks_count", 0),
                "load_time": load_time
            }
            
        except Exception as e:
            logger.error(f"Error loading document {document_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error loading document: {str(e)}",
                "document_id": document_id
            }
    
    def unload_document(self, document_id: str) -> Dict[str, Any]:
        """
        Unload a document from memory.
        
        Args:
            document_id: ID of document to unload
            
        Returns:
            Status dictionary
        """
        if document_id not in self.loaded_documents:
            return {
                "status": "warning",
                "message": f"Document {document_id} is not loaded",
                "document_id": document_id
            }
        
        try:
            # Remove from loaded documents
            self.loaded_documents.remove(document_id)
            
            # Remove from cache
            if document_id in self.document_cache:
                del self.document_cache[document_id]
            
            # Clear chunk cache for this document
            to_remove = []
            for key in self.chunk_cache:
                if key.startswith(f"{document_id}:"):
                    to_remove.append(key)
            
            for key in to_remove:
                del self.chunk_cache[key]
            
            return {
                "status": "success",
                "message": f"Document {document_id} unloaded successfully",
                "document_id": document_id
            }
            
        except Exception as e:
            logger.error(f"Error unloading document {document_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error unloading document: {str(e)}",
                "document_id": document_id
            }
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get chunks for a document (loads from disk if needed).
        
        Args:
            document_id: Document ID
            
        Returns:
            List of document chunks
        """
        # Check if document is loaded
        if document_id not in self.loaded_documents:
            logger.warning(f"Document {document_id} is not loaded")
            self.load_document(document_id)
        
        # Check if chunks are already cached
        chunks_key = f"{document_id}:chunks"
        if chunks_key in self.chunk_cache:
            return self.chunk_cache[chunks_key]
        
        # Get document directory
        if document_id not in self.document_cache:
            logger.warning(f"Document {document_id} not in cache")
            return []
        
        doc_dir = self.document_cache[document_id]["doc_dir"]
        
        # Load chunks from file
        chunks_path = os.path.join(doc_dir, "all_chunks.json")
        if not os.path.exists(chunks_path):
            logger.warning(f"Chunks file not found: {chunks_path}")
            return []
        
        try:
            with open(chunks_path, "r") as f:
                chunks = json.load(f)
            
            # Cache chunks
            self.chunk_cache[chunks_key] = chunks
            
            return chunks
        except Exception as e:
            logger.error(f"Error loading chunks for document {document_id}: {str(e)}")
            return []
    
    def get_document_embeddings(self, document_id: str, chunk_id: str) -> Optional[List[float]]:
        """
        Get embedding for a specific chunk (loads from disk if needed).
        
        Args:
            document_id: Document ID
            chunk_id: Chunk ID
            
        Returns:
            Embedding vector or None if not found
        """
        # Check if document is loaded
        if document_id not in self.loaded_documents:
            logger.warning(f"Document {document_id} is not loaded")
            self.load_document(document_id)
        
        # Check if embedding is already cached
        emb_key = f"{document_id}:{chunk_id}:embedding"
        if emb_key in self.chunk_cache:
            return self.chunk_cache[emb_key]
        
        # Get document directory
        if document_id not in self.document_cache:
            logger.warning(f"Document {document_id} not in cache")
            return None
        
        doc_dir = self.document_cache[document_id]["doc_dir"]
        
        # Try loading individual embedding file
        emb_path = os.path.join(doc_dir, "embeddings", f"{chunk_id}.pkl")
        if os.path.exists(emb_path):
            try:
                with open(emb_path, "rb") as f:
                    embedding = pickle.load(f)
                
                # Cache embedding
                self.chunk_cache[emb_key] = embedding
                
                return embedding
            except Exception as e:
                logger.error(f"Error loading embedding for chunk {chunk_id}: {str(e)}")
        
        # Try loading from all_embeddings file
        all_emb_path = os.path.join(doc_dir, "all_embeddings.pkl")
        if os.path.exists(all_emb_path):
            try:
                with open(all_emb_path, "rb") as f:
                    all_embeddings = pickle.load(f)
                
                if chunk_id in all_embeddings:
                    embedding = all_embeddings[chunk_id]
                    
                    # Cache embedding
                    self.chunk_cache[emb_key] = embedding
                    
                    return embedding
            except Exception as e:
                logger.error(f"Error loading embeddings from all_embeddings: {str(e)}")
        
        return None
    
    def get_document_tree(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get RAPTOR tree structure for a document (loads from disk if needed).
        
        Args:
            document_id: Document ID
            
        Returns:
            Tree structure or None if not found
        """
        # Check if document is loaded
        if document_id not in self.loaded_documents:
            logger.warning(f"Document {document_id} is not loaded")
            self.load_document(document_id)
        
        # Check if tree is already cached
        tree_key = f"{document_id}:tree"
        if tree_key in self.chunk_cache:
            return self.chunk_cache[tree_key]
        
        # Get document directory
        if document_id not in self.document_cache:
            logger.warning(f"Document {document_id} not in cache")
            return None
        
        doc_dir = self.document_cache[document_id]["doc_dir"]
        
        # Load tree from file
        tree_path = os.path.join(doc_dir, "tree_structure.json")
        if not os.path.exists(tree_path):
            logger.warning(f"Tree structure file not found: {tree_path}")
            return None
        
        try:
            with open(tree_path, "r") as f:
                tree = json.load(f)
            
            # Cache tree
            self.chunk_cache[tree_key] = tree
            
            return tree
        except Exception as e:
            logger.error(f"Error loading tree for document {document_id}: {str(e)}")
            return None
    
    def query(
        self, 
        question: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        use_tree: bool = False,
        layer: Optional[int] = None,
        use_history: bool = True
    ) -> Dict[str, Any]:
        """
        Query loaded documents to answer a question.
        
        Args:
            question: User question
            document_ids: Specific document IDs to search (None = all loaded)
            top_k: Number of chunks to retrieve
            use_tree: Whether to use tree-based retrieval
            layer: Specific tree layer to use (None = auto-select)
            use_history: Whether to use chat history for context
            
        Returns:
            Query result with answer and sources
        """
        start_time = time.time()
        
        # Load document IDs if not specified
        if document_ids is None:
            document_ids = list(self.loaded_documents)
        else:
            # Load any documents that aren't already loaded
            for doc_id in document_ids:
                if doc_id not in self.loaded_documents:
                    self.load_document(doc_id)
        
        if not document_ids:
            return {
                "status": "error",
                "message": "No documents specified or loaded",
                "answer": "I don't have any documents loaded to answer your question.",
                "sources": [],
                "time_taken": time.time() - start_time
            }
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(question)
        
        # Retrieve relevant chunks
        if use_tree:
            chunks = self._retrieve_with_tree(
                query_embedding, 
                document_ids, 
                top_k, 
                layer
            )
        else:
            chunks = self._retrieve_flat(
                query_embedding, 
                document_ids, 
                top_k
            )
        
        if not chunks:
            return {
                "status": "warning",
                "message": "No relevant chunks found",
                "answer": "I couldn't find any relevant information in the documents to answer your question.",
                "sources": [],
                "time_taken": time.time() - start_time
            }
        
        # Create context from chunks
        context = "\n\n".join([chunk["content"] for chunk in chunks])
        
        # Get chat history if requested
        chat_history_text = ""
        if use_history:
            chat_history_text = self.chat_history.get_formatted_history()
        
        # Generate answer
        try:
            logging.debug(f'context: {context}, question: {question}, chat_history: {chat_history_text}')
            chain = self.prompt_template | self.llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "question": question,
                "chat_history": chat_history_text
            })
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer = f"Error generating answer: {str(e)}"
        
        # Add interaction to chat history
        self.chat_history.add_interaction(question, answer)
        
        # Format source information
        sources = []
        for chunk in chunks:
            # Extract bbox information if available
            metadata = chunk.get("metadata", {})
            original_boxes = metadata.get("original_boxes", [])
            
            source = {
                "document_id": chunk["document_id"],
                "content": chunk["content"],
                "score": chunk["score"],
                "metadata": metadata,
                "original_boxes": original_boxes  # Preserve original_boxes for highlighting
            }
            sources.append(source)

        time_taken = time.time() - start_time

        return {
            "status": "success",
            "message": f"Query processed in {time_taken:.2f} seconds",
            "answer": answer,
            "sources": sources,
            "time_taken": time_taken
        }
    
    def clear_history(self) -> Dict[str, Any]:
        """
        Clear the chat history.
        
        Returns:
            Status dictionary
        """
        try:
            self.chat_history.clear()
            return {
                "status": "success",
                "message": "Chat history cleared successfully"
            }
        except Exception as e:
            logger.error(f"Error clearing chat history: {str(e)}")
            return {
                "status": "error",
                "message": f"Error clearing chat history: {str(e)}"
            }

    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Get the current chat history.
        
        Returns:
            List of interaction dictionaries
        """
        return self.chat_history.get_history()
    
    def _retrieve_flat(
        self, 
        query_embedding: List[float],
        document_ids: List[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using flat retrieval (no tree).
        
        Args:
            query_embedding: Query embedding vector
            document_ids: Document IDs to search
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with scores
        """
        all_results = []
        
        for doc_id in document_ids:
            # Get document chunks
            chunks = self.get_document_chunks(doc_id)
            
            for chunk in chunks:
                chunk_id = chunk["id"]
                
                # Get embedding for this chunk
                embedding = self.get_document_embeddings(doc_id, chunk_id)
                
                if embedding is None:
                    # Skip if no embedding available
                    continue
                
                # Calculate similarity
                score = self._calculate_similarity(query_embedding, embedding)
                
                # Add to results
                all_results.append({
                    "document_id": doc_id,
                    "chunk_id": chunk_id,
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "score": score
                })
        
        # Sort by score (highest first)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top results
        return all_results[:top_k]
    
    def _retrieve_with_tree(
        self, 
        query_embedding: List[float],
        document_ids: List[str],
        top_k: int,
        start_layer: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using tree-based hierarchical retrieval.
        
        Args:
            query_embedding: Query embedding vector
            document_ids: Document IDs to search
            top_k: Number of chunks to retrieve
            start_layer: Specific tree layer to start from (None = highest)
            
        Returns:
            List of relevant chunks with scores
        """
        all_results = []
        
        for doc_id in document_ids:
            # Get document tree
            tree = self.get_document_tree(doc_id)
            
            if not tree:
                # Skip if no tree available
                logger.warning(f"No tree structure found for document {doc_id}")
                continue
            
            # Get document chunks
            chunks = self.get_document_chunks(doc_id)
            
            if not chunks:
                # Skip if no chunks available
                logger.warning(f"No chunks found for document {doc_id}")
                continue
            
            # Determine start layer if not specified
            if start_layer is None:
                levels = tree.get("levels", [0])
                start_layer = max(levels)
            
            # Get nodes at start layer
            nodes = tree.get("nodes", {})
            layer_nodes = {
                node_id: info for node_id, info in nodes.items() 
                if info.get("level") == start_layer
            }
            
            if not layer_nodes:
                logger.warning(f"No nodes found at layer {start_layer} for document {doc_id}")
                continue
            
            # Score nodes at this layer
            scored_nodes = []
            for node_id, node_info in layer_nodes.items():
                # Generate embedding for this node if not available
                # This is a temporary solution - in a production system,
                # you'd want to use pre-computed embeddings
                node_content = node_info.get("content", "")
                
                # Try to get embedding from cache based on node ID
                emb_key = f"{doc_id}:{node_id}:embedding"
                if emb_key in self.chunk_cache:
                    node_embedding = self.chunk_cache[emb_key]
                else:
                    # Generate on-the-fly
                    node_embedding = self.embeddings.embed_query(node_content)
                    # Cache for future use
                    self.chunk_cache[emb_key] = node_embedding
                
                # Calculate similarity
                score = self._calculate_similarity(query_embedding, node_embedding)
                
                # Add to scored nodes
                scored_nodes.append({
                    "node_id": node_id,
                    "score": score,
                    "children": node_info.get("children", [])
                })
            
            # Sort by score (highest first)
            scored_nodes.sort(key=lambda x: x["score"], reverse=True)
            
            # Get top nodes
            top_nodes = scored_nodes[:min(top_k, len(scored_nodes))]
            
            # Traverse to leaf nodes
            for node in top_nodes:
                leaf_nodes = self._get_leaf_nodes(node["node_id"], nodes)
                
                for leaf_id in leaf_nodes:
                    # Get chunk index from leaf ID
                    match = re.match(r"leaf_(\d+)", leaf_id)
                    if match:
                        idx = int(match.group(1))
                        if idx < len(chunks):
                            chunk = chunks[idx]
                            
                            # Add to results
                            all_results.append({
                                "document_id": doc_id,
                                "chunk_id": chunk["id"],
                                "content": chunk["content"],
                                "metadata": chunk["metadata"],
                                "score": node["score"] * 0.9  # Slight penalty for descendant
                            })
        
        # Sort by score (highest first) and deduplicate
        seen_chunks = set()
        filtered_results = []
        
        for result in sorted(all_results, key=lambda x: x["score"], reverse=True):
            chunk_key = f"{result['document_id']}:{result['chunk_id']}"
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                filtered_results.append(result)
                
                if len(filtered_results) >= top_k:
                    break
        
        return filtered_results
    
    def _get_leaf_nodes(self, node_id: str, nodes: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Get all leaf node descendants of a node.
        
        Args:
            node_id: Starting node ID
            nodes: Dictionary of all nodes in the tree
            
        Returns:
            List of leaf node IDs
        """
        if node_id not in nodes:
            return []
        
        # If this is already a leaf node, return it
        if node_id.startswith("leaf_"):
            return [node_id]
        
        # Get children
        children = nodes[node_id].get("children", [])
        
        if not children:
            return []
        
        # Recursively find all leaf descendants
        leaf_nodes = []
        for child_id in children:
            leaf_nodes.extend(self._get_leaf_nodes(child_id, nodes))
        
        return leaf_nodes
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Convert to NumPy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure the result is within bounds
        return max(0.0, min(1.0, similarity))
    
    def cleanup(self):
        """Clean up resources."""
        # Clear document cache
        self.document_cache.clear()
        self.chunk_cache.clear()
        self.loaded_documents.clear()
        
        # Remove temporary cache directory if we created one
        if os.path.exists(self.cache_dir) and "doc_query_cache" in self.cache_dir:
            shutil.rmtree(self.cache_dir)


def format_document_table(documents: List[Dict[str, Any]]) -> str:
    """
    Format document list as a nicely formatted table.
    
    Args:
        documents: List of document data
        
    Returns:
        Formatted table string
    """
    if not documents:
        return "No documents available."
    
    # Prepare table data
    table_data = []
    
    for doc in documents:
        # Get status icon
        status = "✅" if doc.get("status") == "processed" else "❌"
        
        # Get loaded status icon
        loaded = "📂" if doc.get("is_loaded", False) else "📁"
        
        # Format content types
        content_types = doc.get("content_types", {})
        if content_types:
            types_str = ", ".join([f"{count} {type}" for type, count in content_types.items()])
        else:
            types_str = "N/A"
        
        # Add row to table
        table_data.append([
            f"{status} {loaded}",
            doc.get("document_id", "Unknown"),
            doc.get("filename", "Unknown"),
            doc.get("chunks_count", 0),
            types_str,
            doc.get("processing_date", "")[:10]  # Just show date portion
        ])
    
    # Create table with headers
    headers = ["Status", "Document ID", "Filename", "Chunks", "Content Types", "Date"]
    
    # Return formatted table
    return tabulate(table_data, headers=headers, tablefmt="grid")

def format_history_display(history_items):
    """
    Format chat history for nice display in CLI.
    
    Args:
        history_items: List of history items (interactions and summaries)
        
    Returns:
        Formatted string for display
    """
    if not history_items:
        return "No chat history available."
    
    formatted = "\n💬 Chat History:\n"
    formatted += "=" * 40 + "\n"
    
    for i, item in enumerate(history_items, 1):
        if item.get("type") == "summary":
            formatted += f"📝 SUMMARY [{item['original_count']} interactions]: \n"
            formatted += f"   {item['content']}\n"
            formatted += "-" * 40 + "\n"
        else:
            formatted += f"Q{i}: {item['question']}\n"
            
            # Truncate long answers for display
            answer = item['answer']
            if len(answer) > 100:
                answer = answer[:100] + "..."
            formatted += f"A{i}: {answer}\n"
            formatted += "-" * 40 + "\n"
    
    return formatted

def run_improved_query_interface():
    """Run the improved document query interface"""
    print("\n📚 Improved Document Query Interface")
    print("==================================")
    print("This interface allows you to browse, load, and query documents from disk storage.")
    print("\nAvailable commands:")
    print("  list                   - List all available documents")
    print("  load <doc_id>          - Load a document")
    print("  unload <doc_id>        - Unload a document")
    print("  loaded                 - Show currently loaded documents")
    print("  info <doc_id>          - Show document details")
    print("  query <question>       - Query using flat retrieval")
    print("  query-tree <question>  - Query using tree-based retrieval")
    print("  history                - Show chat history")
    print("  clear-history          - Clear chat history")
    print("  query-no-history <q>   - Query without using chat history")
    print("  exit                   - Exit the interface")
    print("==================================")
    
    # Initialize the query interface
    querier = DiskBasedDocumentQuerier()
    
    # Show initial document list
    docs = querier.list_available_documents()
    if docs:
        print("\n📋 Available Documents:")
        print(format_document_table(docs))
    else:
        print("\n⚠️ No documents available in storage.")
        print("Upload documents using the document uploader before querying.")
    
    # Main interaction loop
    while True:
        try:
            user_input = input("\n🔍 > ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == 'exit':
                print("Cleaning up resources...")
                querier.cleanup()
                print("Exiting query interface. Goodbye!")
                break
            
            elif user_input.lower() == 'list':
                docs = querier.list_available_documents()
                if docs:
                    print("\n📋 Available Documents:")
                    print(format_document_table(docs))
                else:
                    print("No documents available.")
            
            elif user_input.lower() == 'loaded':
                if not querier.loaded_documents:
                    print("No documents currently loaded.")
                else:
                    print("\n📂 Currently Loaded Documents:")
                    for i, doc_id in enumerate(querier.loaded_documents, 1):
                        metadata = querier.document_metadata.get(doc_id, {})
                        filename = metadata.get("original_filename", "Unknown")
                        chunks = metadata.get("chunks_count", 0)
                        print(f"  {i}. {doc_id} - {filename} ({chunks} chunks)")
            
            elif user_input.lower().startswith('load '):
                doc_id = user_input[5:].strip()
                print(f"Loading document: {doc_id}...")
                result = querier.load_document(doc_id)
                
                if result["status"] == "success":
                    print(f"✅ {result['message']} in {result.get('load_time', 0):.2f} seconds")
                    if "chunks_count" in result:
                        print(f"   Chunks: {result['chunks_count']}")
                else:
                    print(f"❌ {result['message']}")
            
            elif user_input.lower().startswith('unload '):
                doc_id = user_input[7:].strip()
                result = querier.unload_document(doc_id)
                print(f"{result['message']}")
            
            elif user_input.lower().startswith('info '):
                doc_id = user_input[5:].strip()
                
                # Load document if not already loaded
                if doc_id not in querier.loaded_documents:
                    print(f"Document {doc_id} is not loaded. Loading now...")
                    querier.load_document(doc_id)
                
                if doc_id in querier.document_metadata:
                    metadata = querier.document_metadata[doc_id]
                    
                    print(f"\n📄 Document Information: {doc_id}")
                    print("=" * (24 + len(doc_id)))
                    print(f"Filename:       {metadata.get('original_filename', 'Unknown')}")
                    print(f"Status:         {metadata.get('status', 'Unknown')}")
                    print(f"Processing Date: {metadata.get('processing_date', 'Unknown')[:19]}")
                    print(f"Processing Time: {metadata.get('processing_time', 0):.2f} seconds")
                    print(f"Chunks:         {metadata.get('chunks_count', 0)}")
                    
                    # Display content types
                    content_types = metadata.get('content_types', {})
                    if content_types:
                        print("\nContent Types:")
                        for content_type, count in content_types.items():
                            print(f"  - {content_type}: {count}")
                    
                    # Display RAPTOR levels
                    raptor_levels = metadata.get('raptor_levels', [])
                    if raptor_levels:
                        print("\nRAPTOR Levels:")
                        for level in sorted(raptor_levels):
                            print(f"  - Level {level}")
                    
                    # Tree structure information
                    tree = querier.get_document_tree(doc_id)
                    if tree:
                        levels = tree.get("levels", [])
                        nodes = tree.get("nodes", {})
                        
                        print("\nTree Structure:")
                        for level in sorted(levels):
                            level_nodes = [n for n_id, n in nodes.items() if n.get("level") == level]
                            level_type = "Original chunks" if level == 0 else "Summary nodes"
                            print(f"  - Level {level}: {len(level_nodes)} nodes ({level_type})")
                else:
                    print(f"No information available for document {doc_id}")
            
            elif user_input.lower().startswith(('query ', 'q ')):
                query = user_input.split(' ', 1)[1].strip()
                
                if not query:
                    print("Please provide a question to query.")
                    continue
                
                if not querier.loaded_documents:
                    print("No documents loaded. Please load at least one document first.")
                    continue
                
                print(f"\n🔍 Processing query (with chat history): {query}")
                result = querier.query(
                    question=query,
                    use_tree=False,
                    top_k=5,
                    use_history=True
                )
                
                print(f"\n✨ Answer ({result['time_taken']:.2f}s):")
                print("=" * 40)
                print(result["answer"])
                print("=" * 40)
                
                print("\n📚 Sources:")
                for i, source in enumerate(result["sources"], 1):
                    print(f"\n{i}. [Score: {source['score']:.4f}] {source['document_id']}")
                    
                    content = source["content"]
                    if len(content) > 300:
                        content = content[:300] + "..."
                    print(f"   {content}")
                    
                    # Show metadata
                    if "metadata" in source and source["metadata"]:
                        metadata = source["metadata"]
                        if "page_idx" in metadata:
                            print(f"   Page: {metadata['page_idx']}")
                        if "type" in metadata:
                            print(f"   Type: {metadata['type']}")
            
            elif user_input.lower().startswith(('query-no-history ', 'qnh ')):
                query = user_input.split(' ', 1)[1].strip()
                
                if not query:
                    print("Please provide a question to query.")
                    continue
                
                if not querier.loaded_documents:
                    print("No documents loaded. Please load at least one document first.")
                    continue
                
                print(f"\n🔍 Processing query (without chat history): {query}")
                result = querier.query(
                    question=query,
                    use_tree=False,
                    top_k=5,
                    use_history=False
                )
                
                print(f"\n✨ Answer ({result['time_taken']:.2f}s):")
                print("=" * 40)
                print(result["answer"])
                print("=" * 40)
                
                print("\n📚 Sources:")
                for i, source in enumerate(result["sources"], 1):
                    print(f"\n{i}. [Score: {source['score']:.4f}] {source['document_id']}")
                    
                    content = source["content"]
                    if len(content) > 300:
                        content = content[:300] + "..."
                    print(f"   {content}")
            
            elif user_input.lower().startswith(('query-tree ', 'qt ')):
                query = user_input.split(' ', 1)[1].strip()
                
                if not query:
                    print("Please provide a question to query.")
                    continue
                
                if not querier.loaded_documents:
                    print("No documents loaded. Please load at least one document first.")
                    continue
                
                print(f"\n🔍 Processing query (tree-based retrieval with chat history): {query}")
                result = querier.query(
                    question=query,
                    use_tree=True,
                    top_k=5,
                    use_history=True
                )
                
                print(f"\n✨ Answer ({result['time_taken']:.2f}s):")
                print("=" * 40)
                print(result["answer"])
                print("=" * 40)
                
                print("\n📚 Sources:")
                for i, source in enumerate(result["sources"], 1):
                    print(f"\n{i}. [Score: {source['score']:.4f}] {source['document_id']}")
                    
                    content = source["content"]
                    if len(content) > 300:
                        content = content[:300] + "..."
                    print(f"   {content}")
            
            elif user_input.lower() == 'history':
                history = querier.get_chat_history()
                if not history:
                    print("No chat history available.")
                else:
                    print(format_history_display(history))
            
            elif user_input.lower() == 'clear-history':
                result = querier.clear_history()
                print(result["message"])
            
            elif user_input.lower() in ['help', '?']:
                print("\nAvailable commands:")
                print("  list                   - List all available documents")
                print("  load <doc_id>          - Load a document")
                print("  unload <doc_id>        - Unload a document")
                print("  loaded                 - Show currently loaded documents")
                print("  info <doc_id>          - Show document details")
                print("  query <question>       - Query using flat retrieval")
                print("  query-tree <question>  - Query using tree-based retrieval")
                print("  history                - Show chat history")
                print("  clear-history          - Clear chat history")
                print("  query-no-history <q>   - Query without using chat history")
                print("  exit                   - Exit the interface")
            
            else:
                print(f"Unknown command: {user_input}")
                print("Type 'help' to see available commands.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    run_improved_query_interface()