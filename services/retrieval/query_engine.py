import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

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
    Supports both flat and hierarchical retrieval methods.
    """
    
    def __init__(
        self,
        config=None,
        vector_store=None,
        embeddings=None,
        llm=None
    ):
        """
        Initialize the query engine with required components.
        
        Args:
            config: Configuration object
            vector_store: Optional VectorStoreAdapter instance
            embeddings: Optional embeddings model
            llm: Optional language model
        """
        self.config = config or get_config()
        
        # Initialize vector store
        self.vector_store = vector_store or VectorStoreAdapter(config=self.config)
        
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
        
        logger.info("Query engine initialized")
    
    def query(
        self,
        question: str,
        document_ids: List[str],
        case_id: str,
        chat_id: None,
        user_id: None,
        use_tree: bool = False,
        top_k: int = 5,
        tree_level_filter: Optional[List[int]] = None,  # None = all levels, [0] = original chunks, [1,2,3] = specific summary levels
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
            top_k: Number of chunks to retrieve
            tree_level_filter: Filter by tree level (None=all, [0]=original chunks only, [1,2,3]=summaries)
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
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(question)
        
        # Retrieve relevant chunks
        retrieval_start = time.time()
        
        # Convert case_id to list format required by vector_store.search
        case_ids = [case_id]
        
        # Set up chunk_types filter if tree_level_filter is specified
        chunk_types = None
        if tree_level_filter is not None:
            if 0 in tree_level_filter and len(tree_level_filter) == 1:
                # Filter for original chunks only
                chunk_types = ["original"]
            elif 0 not in tree_level_filter and tree_level_filter:
                # Filter for summary chunks only
                chunk_types = ["summary"]
        
        # Perform search directly with all filters
        chunks = self.vector_store.search(
            query_embedding=query_embedding,
            case_ids=case_ids,
            document_ids=document_ids,
            tree_levels=tree_level_filter,
            chunk_types=chunk_types,
            top_k=top_k
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
            
            # Add chunk to context
            context_parts.append(f"{header}\n{content}\n")
        
        return "\n\n".join(context_parts)
    
    def retrieve_relevant_chunks(
        self,
        query_embedding: List[float],
        case_id: str,
        document_ids: List[str],
        use_tree: bool = False,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks based on query embedding.
        
        Args:
            query_embedding: Query embedding vector
            case_id: Case ID
            document_ids: List of document IDs to search
            use_tree: Whether to use tree-based retrieval
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks
        """
        # Case ID should be provided as a list for the vector store
        case_ids = [case_id]
        
        if use_tree:
            # For tree-based retrieval, first get higher level nodes
            content_types = None
            
            # First search for summary nodes (tree nodes)
            chunks = self.vector_store.search(
                query_embedding=query_embedding,
                case_ids=case_ids,
                document_ids=document_ids,
                content_types=content_types,
                chunk_types=["summary"],
                top_k=top_k
            )
            
            # If we found summary nodes, get their children
            if chunks:
                logger.info(f"Found {len(chunks)} summary nodes for hierarchical retrieval")
                
                # Get original chunks (leaf nodes) related to the summaries
                # In a more sophisticated implementation, we would follow the actual
                # tree structure down to leaf nodes, but for now we'll just retrieve
                # original chunks with a new query
                original_chunks = self.vector_store.search(
                    query_embedding=query_embedding,
                    case_ids=case_ids,
                    document_ids=document_ids,
                    content_types=content_types,
                    chunk_types=["original"],
                    top_k=top_k
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
                return self._flat_retrieval(query_embedding, case_ids, document_ids, top_k)
        else:
            # Flat retrieval - just get original chunks
            return self._flat_retrieval(query_embedding, case_ids, document_ids, top_k)
    
    def _flat_retrieval(
        self, 
        query_embedding: List[float],
        case_ids: List[str],
        document_ids: List[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Perform flat retrieval of original chunks.
        
        Args:
            query_embedding: Query embedding vector
            case_ids: List of case IDs
            document_ids: List of document IDs
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks
        """
        return self.vector_store.search(
            query_embedding=query_embedding,
            case_ids=case_ids,
            document_ids=document_ids,
            chunk_types=["original"],
            top_k=top_k
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