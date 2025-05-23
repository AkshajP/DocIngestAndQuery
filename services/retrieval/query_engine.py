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
        use_tree: bool = False,
        use_hybrid: Optional[bool] = None,  # New parameter
        vector_weight: Optional[float] = None,  # New parameter
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
        query_text: Optional[str] = None,  # New parameter for BM25
        use_tree: bool = False,
        use_hybrid: bool = False,  # New parameter
        vector_weight: float = 0.5,  # New parameter
        top_k: int = 5,
        tree_level_filter: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using vector similarity or hybrid search.
        
        Args:
            query_embedding: Query embedding vector
            case_id: Case ID
            document_ids: List of document IDs to search
            query_text: Original query text for BM25 search
            use_tree: Whether to use tree-based retrieval
            use_hybrid: Whether to use hybrid search
            vector_weight: Weight for vector scores (0-1)
            top_k: Number of chunks to retrieve
            tree_level_filter: Filter by tree level
            
        Returns:
            List of relevant chunks
        """
        # Case ID should be provided as a list for the vector store
        case_ids = [case_id]
        
        if use_hybrid and not query_text:
            logger.warning("Hybrid search requested but no query_text provided. Using vector-only search.")
            use_hybrid = False
        
        if use_tree:
            # Enhanced tree-based retrieval with hybrid search
            # First find relevant summary nodes, potentially using hybrid search
            content_types = None
            chunk_types = None
            
            # Set up chunk_types filter if tree_level_filter is specified
            if tree_level_filter is not None:
                if 0 in tree_level_filter and len(tree_level_filter) == 1:
                    # Filter for original chunks only
                    chunk_types = ["original"]
                elif 0 not in tree_level_filter and tree_level_filter:
                    # Filter for summary chunks only
                    chunk_types = ["summary"]
            
            # Search for summary nodes, potentially with hybrid search
            chunks = self.vector_store.search(
                query_embedding=query_embedding,
                case_ids=case_ids,
                document_ids=document_ids,
                content_types=["text", "table"],
                chunk_types=chunk_types,
                tree_levels=tree_level_filter,
                top_k=top_k,
                use_hybrid=use_hybrid,
                vector_weight=vector_weight,
                query_text=query_text
            )
            
            # If we found summary nodes, get their children
            if chunks:
                logger.info(f"Found {len(chunks)} summary nodes for hierarchical retrieval")
                
                # Get original chunks related to the summaries
                # Use hybrid search here too if enabled
                original_chunks = self.vector_store.search(
                    query_embedding=query_embedding,
                    case_ids=case_ids,
                    document_ids=document_ids,
                    content_types=content_types,
                    chunk_types=["original"],
                    top_k=top_k,
                    use_hybrid=use_hybrid,
                    vector_weight=vector_weight,
                    query_text=query_text
                )
                
                # Combine chunks, prioritizing original content
                # We'll give a slight boost to original chunks in the ranking
                all_chunks = original_chunks + chunks
                
                # Sort by score (highest first) and deduplicate if needed
                seen_content = set()
                filtered_chunks = []
                
                for chunk in sorted(all_chunks, key=lambda x: x.get("score", 0), reverse=True):
                    # Use a short hash of content to deduplicate
                    content_hash = hash(chunk.get("content", "")[:100])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        filtered_chunks.append(chunk)
                        
                        if len(filtered_chunks) >= top_k:
                            break
                
                return filtered_chunks
            else:
                # Fall back to flat retrieval if no summary nodes found
                logger.info("No summary nodes found, falling back to flat retrieval")
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
        else:
            # Flat retrieval - just get original chunks
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
    
    def _flat_retrieval(
        self, 
        query_embedding: List[float],
        case_ids: List[str],
        document_ids: List[str],
        top_k: int,
        query_text: Optional[str] = None,  # New parameter
        tree_level_filter: Optional[List[int]] = None,
        use_hybrid: bool = False,  # New parameter
        vector_weight: float = 0.5  # New parameter
    ) -> List[Dict[str, Any]]:
        """
        Perform flat retrieval of original chunks.
        
        Args:
            query_embedding: Query embedding vector
            case_ids: List of case IDs
            document_ids: List of document IDs
            top_k: Number of chunks to retrieve
            query_text: Original query text for BM25 search
            tree_level_filter: Optional filter by tree level
            use_hybrid: Whether to use hybrid search
            vector_weight: Weight for vector scores (0-1)
            
        Returns:
            List of relevant chunks
        """
        # Set up chunk_types filter based on tree_level_filter
        chunk_types = None
        if tree_level_filter is not None:
            if 0 in tree_level_filter and len(tree_level_filter) == 1:
                # Filter for original chunks only
                chunk_types = ["original"]
            elif 0 not in tree_level_filter and tree_level_filter:
                # Filter for summary chunks only
                chunk_types = ["summary"]
                
        return self.vector_store.search(
            query_embedding=query_embedding,
            case_ids=case_ids,
            document_ids=document_ids,
            content_types=["text", "table"],
            chunk_types=chunk_types,
            tree_levels=tree_level_filter,
            top_k=top_k,
            use_hybrid=use_hybrid,
            vector_weight=vector_weight,
            query_text=query_text
        )
    
    def process_with_llm(
        self,
        question: str,
        context: str,
        chat_history: str,
        model: Optional[str] = None
    ) -> Tuple[str, int, str]:
        """
        Process the query context with an LLM to generate an answer.
        
        Args:
            question: User question
            context: Context from retrieved chunks
            chat_history: Formatted chat history text
            model: Optional model override
            
        Returns:
            Tuple of (answer, token_count, model_used)
        """
        # Select LLM based on preference if provided
        llm = self.llm
        if model and model != self.config.ollama.model:
            try:
                llm = OllamaLLM(
                    model=model,
                    base_url=self.config.ollama.base_url,
                    temperature=0.1
                )
                model_used = model
            except Exception as e:
                logger.error(f"Error initializing preferred model {model}: {str(e)}")
                logger.info(f"Falling back to default model: {self.config.ollama.model}")
                model_used = self.config.ollama.model
        else:
            model_used = self.config.ollama.model
        
        # Estimate token usage (approximate)
        token_count = self.estimate_token_usage(context, question, chat_history)
        
        # Generate answer
        try:
            # First, get the full prompt without executing it
            full_prompt = self.query_prompt.format(
                context=context,
                question=question,
                chat_history=chat_history
            )
            
            # Log the complete prompt
            logger.info("============= FULL PROMPT SENT TO LLM =============")
            logger.info(f"Question: {question}")
            logger.info(f"Model: {model_used}")
            logger.info(f"Estimated tokens: {token_count}")
            
            # Log the chat history section
            if chat_history:
                logger.info("--------------- CHAT HISTORY SECTION ---------------")
                logger.info(chat_history)
            else:
                logger.info("--------------- NO CHAT HISTORY PROVIDED ---------------")
                
            # Log the context section (truncated if very long)
            logger.info("--------------- CONTEXT SECTION ---------------")
            if len(context) > 1000:
                logger.info(f"{context[:1000]}... (truncated, total length: {len(context)} chars)")
            else:
                logger.info(context)
                
            # Log the full prompt if debug is enabled
            if self.config.debug:
                logger.debug("--------------- COMPLETE PROMPT ---------------")
                logger.debug(full_prompt)
                
            logger.info("====================================================")
            
            # Now execute the chain
            chain = self.prompt_template | llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "question": question,
                "chat_history": chat_history
            })
            
            return answer, token_count, model_used
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
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