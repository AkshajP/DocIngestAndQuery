import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from typing import AsyncGenerator
import json
import requests
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.config import get_config
from db.document_store.repository import DocumentMetadataRepository
from db.vector_store.adapter import VectorStoreAdapter
from services.ml.embeddings import EmbeddingService
from .chunk_deduplicator import ChunkDeduplicator

logger = logging.getLogger(__name__)

class QueryEngine:
    """
    Query engine for retrieving and processing document content.
    Supports both flat and hierarchical retrieval methods with hybrid search.
    """
    
    def __init__(
        self,
        config=None,
        vector_store=None,
        embeddings=None,
        llm=None
    ):
        """Initialize the query engine with required components."""
        self.config = config or get_config()
        
        # Initialize vector store with hybrid search enabled by default
        self.vector_store = vector_store or VectorStoreAdapter(
            config=self.config,
            hybrid_search_enabled=True  # Enable hybrid search
        )
        
        # Store hybrid search configuration
        self.hybrid_search_enabled = self.vector_store.supports_hybrid_search()
        self.default_vector_weight = self.config.vector_db.default_vector_weight
        
        # Initialize embeddings model
        self.embeddings = embeddings or OllamaEmbeddings(
            model=self.config.ollama.embed_model,
            base_url=self.config.ollama.base_url
        )
        
        # Initialize LLM
        self.llm = llm or OllamaLLM(
            model=self.config.ollama.model,
            base_url=self.config.ollama.base_url,
            temperature=self.config.ollama.temperature
        )
        self.chunk_deduplicator = ChunkDeduplicator()
        
        # Create RAG prompt template
        self.query_prompt = PromptTemplate(
            template="""
                <system>
                You are a document analysis assistant. Answer the user's question based on the information from the documents I've indexed for you and the previous conversation.
                Only use information from the documents shown below and refer to previous conversations if needed. If the answer cannot be formed satisfactorily, acknowledge that you don't have enough information.
                Never claim the user wrote or said anything that appears in the documents.
                </system>

                <user_query>
                {question}
                </user_query>
                
                <previous_conversation>
                {chat_history}
                </previous_conversation>

                <indexed_documents>
                {context}
                </indexed_documents>

                <assistant>
                """,
            input_variables=["context", "question", "chat_history"]
        )
        
        logger.info(f"Query engine initialized with hybrid search: {self.hybrid_search_enabled}")
    
    def query(
        self,
        question: str,
        document_ids: List[str],
        case_id: str,
        chat_id: None,
        user_id: None,
        chat_settings: Optional[Any] = None,  # Add this parameter
        use_tree: bool = False,
        use_hybrid: Optional[bool] = None,
        vector_weight: Optional[float] = None,
        top_k: int = 10,
        tree_level_filter: Optional[List[int]] = None,
        chat_history: str = "",
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a query against the document collection.
        
        Args:
            question: User question
            document_ids: List of document IDs to search
            case_id: Case ID for filtering
            chat_settings: Chat-specific settings that override other parameters
            use_tree: Whether to use tree-based retrieval
            use_hybrid: Whether to use hybrid search (None = auto-detect)
            vector_weight: Weight for vector scores (0-1, None = use default)
            top_k: Number of chunks to retrieve
            tree_level_filter: Filter by tree level
            chat_history: Previous conversation history
            model_override: Optional model to use instead of default
            
        Returns:
            Query result with answer and sources
        """
        
        # Apply chat settings if provided, with fallback to passed parameters
        if chat_settings:
            use_tree = getattr(chat_settings, 'use_tree_search', use_tree)
            use_hybrid = getattr(chat_settings, 'use_hybrid_search', use_hybrid)
            vector_weight = getattr(chat_settings, 'vector_weight', vector_weight)
            top_k = getattr(chat_settings, 'top_k', top_k)
            tree_level_filter = getattr(chat_settings, 'tree_level_filter', tree_level_filter)
            model_override = getattr(chat_settings, 'llm_model', model_override) or model_override
        
        if not document_ids:
            return {
                "status": "error",
                "message": "No documents specified",
                "answer": "I don't have any documents to answer your question.",
                "sources": []
            }
        
        start_time = time.time()
        
        # Determine whether to use hybrid search
        should_use_hybrid = use_hybrid
        if should_use_hybrid is None:
            # Auto-detect based on configuration and availability
            should_use_hybrid = self.hybrid_search_enabled
        
        # Use default vector weight if not specified
        if vector_weight is None:
            vector_weight = self.default_vector_weight
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(question)
        
        # Retrieve relevant chunks with hybrid search support
        retrieval_start = time.time()
        
        chunks = self.retrieve_relevant_chunks(
            query_embedding=query_embedding,
            query_text=question,  # Pass original question for BM25
            case_id=case_id,
            document_ids=document_ids,
            use_tree=use_tree,
            use_hybrid=should_use_hybrid,
            vector_weight=vector_weight,
            top_k=top_k,
            tree_level_filter=tree_level_filter
        )
        
        retrieval_time = time.time() - retrieval_start
        
        if not chunks:
            return {
                "status": "warning",
                "message": "No relevant chunks found",
                "answer": "I couldn't find any relevant information in the documents to answer your question.",
                "sources": [],
                "time_taken": time.time() - start_time,
                "retrieval_time": retrieval_time,
                "llm_time": 0
            }
        
        # Create context from chunks
        context = self._build_context_from_chunks(chunks)
        
        # Generate answer
        llm_start = time.time()
        try:
            # Estimate token counts
            input_token_count = self.estimate_token_count(context + question + chat_history)
            
            # Customize model based on override if provided
            current_llm = self.llm
            if model_override:
                current_llm = OllamaLLM(
                    model=model_override,
                    base_url=self.config.ollama.base_url,
                    temperature=self.config.ollama.temperature
                )
            
            # Generate answer
            chain = self.query_prompt | current_llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "question": question,
                "chat_history": chat_history
            })
            
            # Roughly estimate output tokens
            output_token_count = len(answer.split()) * 1.35  # Very rough approximation
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "status": "error",
                "message": f"Error generating answer: {str(e)}",
                "answer": "I encountered an error while trying to answer your question.",
                "sources": [],
                "time_taken": time.time() - start_time,
                "retrieval_time": retrieval_time,
                "llm_time": time.time() - llm_start
            }
        
        llm_time = time.time() - llm_start
        
        # Format source information
        sources = []
        for chunk in chunks:
            source = {
                "document_id": chunk["document_id"],
                "content": chunk["content"],
                "score": chunk["score"],
                "metadata": chunk.get("metadata", {}),
                "tree_level": chunk.get("tree_level", 0)
            }
            sources.append(source)

        time_taken = time.time() - start_time

        return {
            "status": "success",
            "message": f"Query processed in {time_taken:.2f} seconds",
            "answer": answer,
            "sources": sources,
            "time_taken": time_taken,
            "retrieval_time": retrieval_time,
            "search_method": "hybrid" if should_use_hybrid else ("tree" if use_tree else "flat"),
            "vector_weight": vector_weight if should_use_hybrid else None,
            "llm_time": llm_time,
            "input_tokens": int(input_token_count),
            "output_tokens": int(output_token_count),
            "model_used": model_override or self.config.ollama.model
        }
        
    def estimate_token_count(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        This is a very rough approximation based on whitespace tokenization.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # This is a very rough approximation
        # For more accurate counts, consider using tiktoken or similar
        words = text.split()
        
        # Roughly 100 words = 75 tokens for typical English text
        return int(len(words) * 0.75)
    
    def _build_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks.
        
        Args:
            chunks: List of chunk objects with content and metadata
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            # Get content and metadata
            content = chunk["content"]
            metadata = chunk.get("metadata", {})
            document_id = chunk["document_id"]
            tree_level = chunk.get("tree_level", 0)
            search_method = chunk.get("search_method", "vector")
            
            # Format based on tree level
            if tree_level > 0:
                # Add summary marker
                header = f"[SUMMARY LEVEL {tree_level} | DOC: {document_id}]"
            else:
                # Add document and page marker if available
                page_num = metadata.get("page_idx", "unknown")
                if isinstance(page_num, int):
                    page_num += 1  # Convert from 0-indexed to 1-indexed for display
                header = f"[DOCUMENT: {document_id} | PAGE: {page_num}]"
            
            # For hybrid search results, include the search method that found this chunk
            if search_method == "hybrid":
                # If we have detailed scores, include them
                vector_score = chunk.get("vector_score", None)
                bm25_score = chunk.get("bm25_score", None)
                
                if vector_score is not None and bm25_score is not None:
                    header += f" [FOUND BY: {search_method.upper()} (V: {vector_score:.2f}, B: {bm25_score:.2f})]"
                else:
                    header += f" [FOUND BY: {search_method.upper()}]"
            elif search_method != "vector":
                header += f" [FOUND BY: {search_method.upper()}]"
            
            # Add chunk to context
            context_parts.append(f"{header}\n{content}\n")
        
        return "\n\n".join(context_parts)
    
    def retrieve_relevant_chunks(
        self,
        query_embedding: List[float],
        case_id: str,
        document_ids: List[str],
        query_text: Optional[str] = None,
        use_tree: bool = False,
        use_hybrid: bool = False,
        vector_weight: float = 0.5,
        top_k: int = 5,
        tree_level_filter: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using vector similarity or hybrid search.
        Enhanced with robust deduplication.
        """
        case_ids = [case_id]
        
        if use_hybrid and not query_text:
            logger.warning("Hybrid search requested but no query_text provided. Using vector-only search.")
            use_hybrid = False
        
        all_chunks = []
        
        if use_tree:
            # Enhanced tree-based retrieval with hybrid search
            all_chunks = self._tree_based_retrieval(
                query_embedding=query_embedding,
                query_text=query_text,
                case_ids=case_ids,
                document_ids=document_ids,
                top_k=top_k,
                tree_level_filter=tree_level_filter,
                use_hybrid=use_hybrid,
                vector_weight=vector_weight
            )
        else:
            # Flat retrieval
            all_chunks = self._flat_retrieval(
                query_embedding=query_embedding,
                query_text=query_text,
                case_ids=case_ids,
                document_ids=document_ids,
                top_k=top_k,
                tree_level_filter=tree_level_filter,
                use_hybrid=use_hybrid,
                vector_weight=vector_weight
            )
        
        # Apply enhanced deduplication
        deduplicated_chunks = self.chunk_deduplicator.deduplicate_chunks(
            chunks=all_chunks,
            top_k=top_k,
            score_merge_strategy="max"  # Use max score when chunks are found via multiple methods
        )
        
        return deduplicated_chunks
    
    def _tree_based_retrieval(
        self,
        query_embedding: List[float],
        query_text: Optional[str],
        case_ids: List[str],
        document_ids: List[str],
        top_k: int,
        tree_level_filter: Optional[List[int]],
        use_hybrid: bool,
        vector_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Enhanced tree-based retrieval that collects chunks from multiple sources
        before deduplication.
        """
        all_chunks = []
        
        # Set up chunk_types filter if tree_level_filter is specified
        summary_chunk_types = None
        original_chunk_types = None
        
        if tree_level_filter is not None:
            if 0 in tree_level_filter and len(tree_level_filter) == 1:
                # Only original chunks requested
                original_chunk_types = ["original"]
                summary_chunk_types = []  # Skip summary search
            elif 0 not in tree_level_filter and tree_level_filter:
                # Only summary chunks requested
                summary_chunk_types = ["summary"]
                original_chunk_types = []  # Skip original search
            else:
                # Mixed levels requested
                summary_chunk_types = ["summary"]
                original_chunk_types = ["original"]
        else:
            # Default: search both summary and original
            summary_chunk_types = ["summary"]
            original_chunk_types = ["original"]
        
        # Search for summary nodes first (if requested)
        if summary_chunk_types:
            summary_chunks = self.vector_store.search(
                query_embedding=query_embedding,
                case_ids=case_ids,
                document_ids=document_ids,
                content_types=["text", "table"],
                chunk_types=summary_chunk_types,
                tree_levels=tree_level_filter,
                top_k=top_k * 2,  # Get more to account for deduplication
                use_hybrid=use_hybrid,
                vector_weight=vector_weight,
                query_text=query_text
            )
            
            # Tag chunks with search context
            for chunk in summary_chunks:
                chunk["search_context"] = "tree_summary"
                if not chunk.get("search_method"):
                    chunk["search_method"] = "hybrid" if use_hybrid else "vector"
            
            all_chunks.extend(summary_chunks)
            logger.info(f"Found {len(summary_chunks)} summary nodes for hierarchical retrieval")
        
        # Search for original chunks (if requested)  
        if original_chunk_types:
            original_chunks = self.vector_store.search(
                query_embedding=query_embedding,
                case_ids=case_ids,
                document_ids=document_ids,
                content_types=["text", "table"],
                chunk_types=original_chunk_types,
                top_k=top_k * 2,  # Get more to account for deduplication
                use_hybrid=use_hybrid,
                vector_weight=vector_weight,
                query_text=query_text
            )
            
            # Tag chunks with search context
            for chunk in original_chunks:
                chunk["search_context"] = "tree_original"
                if not chunk.get("search_method"):
                    chunk["search_method"] = "hybrid" if use_hybrid else "vector"
            
            all_chunks.extend(original_chunks)
            logger.info(f"Found {len(original_chunks)} original chunks for hierarchical retrieval")
        
        # If no chunks found, fall back to flat retrieval
        if not all_chunks:
            logger.info("No chunks found in tree search, falling back to flat retrieval")
            return self._flat_retrieval(
                query_embedding=query_embedding,
                query_text=query_text,
                case_ids=case_ids,
                document_ids=document_ids,
                top_k=top_k,
                tree_level_filter=tree_level_filter,
                use_hybrid=use_hybrid,
                vector_weight=vector_weight
            )
        
        return all_chunks
    
    def _flat_retrieval(
        self, 
        query_embedding: List[float],
        case_ids: List[str],
        document_ids: List[str],
        top_k: int,
        query_text: Optional[str] = None,
        tree_level_filter: Optional[List[int]] = None,
        use_hybrid: bool = False,
        vector_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform flat retrieval of chunks with enhanced metadata tagging.
        """
        # Set up chunk_types filter based on tree_level_filter
        chunk_types = None
        if tree_level_filter is not None:
            if 0 in tree_level_filter and len(tree_level_filter) == 1:
                chunk_types = ["original"]
            elif 0 not in tree_level_filter and tree_level_filter:
                chunk_types = ["summary"]
                
        chunks = self.vector_store.search(
            query_embedding=query_embedding,
            case_ids=case_ids,
            document_ids=document_ids,
            content_types=["text", "table"],
            chunk_types=chunk_types,
            tree_levels=tree_level_filter,
            top_k=top_k * 2,  # Get more to account for potential deduplication
            use_hybrid=use_hybrid,
            vector_weight=vector_weight,
            query_text=query_text
        )
        
        # Tag chunks with search context
        for chunk in chunks:
            chunk["search_context"] = "flat"
            if not chunk.get("search_method"):
                chunk["search_method"] = "hybrid" if use_hybrid else "vector"
        
        return chunks

    def estimate_token_usage(
        self, 
        context: str, 
        question: str, 
        history: str
    ) -> int:
        """
        Estimate token usage for the query.
        
        Args:
            context: Context text
            question: Question text
            history: Chat history text
            
        Returns:
            Estimated token count
        """
        # Very rough estimation - 1 token ≈ 4 characters for English text
        prompt_len = len(self.query_prompt.template)
        context_len = len(context)
        question_len = len(question)
        history_len = len(history)
        
        # Convert characters to tokens (rough approximation)
        total_chars = prompt_len + context_len + question_len + history_len
        estimated_tokens = total_chars // 4
        
        # Add a buffer for special tokens and model specifics
        estimated_tokens = int(estimated_tokens * 1.1)
        
        return estimated_tokens
    
    async def stream_response(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        chat_history: str = "",
        model: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream the response token by token from the LLM.
        
        Args:
            question: User question
            chunks: Retrieved chunks with content and metadata
            chat_history: Formatted chat history text
            model: Optional model override
            
        Yields:
            Response tokens as they are generated
        """
        # Select LLM based on preference if provided
        llm_name = model or self.config.ollama.model
        search_methods = set(chunk.get("search_method", "vector") for chunk in chunks)
        search_method = "hybrid" if "hybrid" in search_methods else "vector"
        
        try:
            # Build context from chunks
            context = self._build_context_from_chunks(chunks)
            
            # Format the prompt
            prompt = self.query_prompt.format(
                context=context,
                question=question,
                chat_history=chat_history
            )
            
            # Log the prompt (for debugging)
            logger.info(f"Streaming response for question: {question}")
            logger.info(f"Using model: {llm_name}")
            
            # Ollama doesn't have a native streaming interface in langchain
            # So we'll use the raw Ollama API through requests
            import requests
            
            # Set up the API call to Ollama
            url = f"{self.config.ollama.base_url}/api/generate"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": llm_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": self.config.ollama.temperature
                }
            }
            
            # Make the streaming API call
            with requests.post(url, json=data, headers=headers, stream=True) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue
                        
                    # Parse the JSON response
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        
                        if token:
                            yield token
                            
                        # Check if done
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode JSON from Ollama: {line}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error streaming response: {str(e)}")
            yield f"Error generating response: {str(e)}"
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format chunks into a context string for the LLM.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            document_id = chunk.get("document_id", "unknown")
            content = chunk.get("content", "")
            score = chunk.get("score", 0)
            
            # Add source information
            source_info = f"[Source {i+1}] Document: {document_id}"
            
            # Add page information if available
            page_number = chunk.get("page_number")
            if page_number is not None:
                source_info += f", Page: {page_number + 1}"  # Convert 0-based to 1-based
            
            # Add content type information
            content_type = chunk.get("content_type", "text")
            if content_type != "text":
                source_info += f", Type: {content_type}"
            
            # Format the chunk with source information
            context_parts.append(f"{source_info}\n\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format chunks into source information for the response.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            List of formatted source dictionaries
        """
        sources = []
        
        for chunk in chunks:
            document_id = chunk.get("document_id", "unknown")
            content = chunk.get("content", "")
            score = chunk.get("score", 0)
            metadata = chunk.get("metadata", {})
            
            # Extract bbox information if available
            original_boxes = metadata.get("original_boxes", [])
            
            source = {
                "document_id": document_id,
                "content": content,
                "score": score,
                "metadata": metadata,
                "content_type": chunk.get("content_type", "text"),
                "page_number": chunk.get("page_number"),
                "original_boxes": original_boxes
            }
            
            sources.append(source)
            
        return sources