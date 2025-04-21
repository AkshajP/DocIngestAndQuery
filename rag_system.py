import os
from typing import List, Dict, Any, Optional
import hashlib

# Document processing
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector database
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# LLM and embeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import Document


class DocumentProcessor:
    """
    
    Handles loading and chunking of documents
    with chunksize 1000 and overlap 200
    
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a document based on file extension"""
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension.lower() in ['.txt', '.md']:
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        documents = loader.load()
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)


class EmbeddingGenerator:
    """Generates embeddings using Ollama"""
    
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=base_url
        )
        self.embedding_dim = None
        
        # Test the embeddings to get dimensions
        try:
            test_embed = self.embed_query("Test embedding dimension")
            self.embedding_dim = len(test_embed)
            print(f"Successfully initialized test embeddings using Ollama with dim: {self.embedding_dim}")
        except Exception as e:
            print(f"Warning: Could not initialize embeddings: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embed = self.embeddings.embed_query(text)
        if self.embedding_dim is None:
            self.embedding_dim = len(embed)
        return embed
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.embed_query(text)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        # Process in smaller batches to avoid potential issues
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            
            # Verify dimensions
            if self.embedding_dim is None and batch_embeddings:
                self.embedding_dim = len(batch_embeddings[0])
                
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings


class MilvusVectorStore:
    """Handles interactions with Milvus vector database"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "document_store",
        dim: int = None  # Will be set based on actual embeddings
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.collection = None
        
        # Connect to Milvus
        connections.connect("default", host=host, port=port)
        
        # Don't initialize collection yet - wait until we know the embedding dimension
    
    def _init_collection(self, dim):
        """Initialize Milvus collection with specified dimension"""
        # Update dimension
        self.dim = dim
        print(f"Initializing collection with dimension: {self.dim}")
        
        # Drop existing collection if it exists
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            
        # Define fields for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        
        # Create collection schema
        schema = CollectionSchema(fields=fields)
        
        # Create collection
        self.collection = Collection(name=self.collection_name, schema=schema, shards_num = 2)
        
        # Create index for vector field
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Add documents and their embeddings to Milvus"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
        
        # Get the dimension from the first embedding
        if not embeddings:
            raise ValueError("No embeddings provided")
            
        embedding_dim = len(embeddings[0])
        print(f"Adding documents with embedding dimension: {embedding_dim}")
        
        # Initialize or reinitialize the collection with the correct dimension
        if self.collection is None or self.dim != embedding_dim:
            self._init_collection(embedding_dim)
        
        # Prepare data for insertion
        ids = []
        contents = []
        metadatas = []
        
        for doc in documents:
            # Create a unique ID based on content hash
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
            ids.append(doc_id)
            contents.append(doc.page_content)
            metadatas.append(doc.metadata)
        
        batch_size = 1000  # Adjust based on your memory constraints
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            # Insert data batch
            entities = [
                ids[i:end_idx],
                contents[i:end_idx],
                metadatas[i:end_idx],
                embeddings[i:end_idx]
            ]
            self.collection.insert(entities)
        
        self.collection.flush()
    
    def similarity_search(self, query_embedding: List[float], metadata_filter: dict = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity"""
        if self.collection is None:
            raise ValueError("Collection not initialized - no documents have been added yet")
            
        # Load collection
        self.collection.load()
        
        # Search parameters
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        
        expr = None
        if metadata_filter:
            conditions = []
            for key, value in metadata_filter.items():
                if isinstance(value, str):
                    conditions.append(f'metadata["{key}"] == "{value}"')
                else:
                    conditions.append(f'metadata["{key}"] == {value}')
            
            if conditions:
                expr = " && ".join(conditions)
        
        # Execute search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["content", "metadata"]
        )

        # Format results
        documents = []
        for hits in results:
            for hit in hits:
                documents.append({
                    "content": hit.entity.get("content"),
                    "metadata": hit.entity.get("metadata"),
                    "score": hit.score
                })
        
        return documents
    
    def release(self):
        """Release collection from memory"""
        if self.collection:
            self.collection.release()


class EnhancedMilvusVectorStore:
    """
    Enhanced version of MilvusVectorStore that handles large text chunks.
    This is a wrapper/extension class that doesn't modify the original.
    """
    
    def __init__(self, milvus_store):
        """Initialize with the original MilvusVectorStore instance."""
        self.milvus_store = milvus_store
        self.MAX_CONTENT_LENGTH = 65000  # Just under Milvus limit of 65535
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """
        Add documents and their embeddings to Milvus, handling large text contents.
        
        Args:
            documents: List of Document objects
            embeddings: List of embedding vectors
        """
        # Process documents to handle content length limits
        processed_documents = []
        processed_embeddings = []
        
        for i, doc in enumerate(documents):
            content = doc.page_content
            
            # Check if content exceeds Milvus limit
            if len(content) > self.MAX_CONTENT_LENGTH:
                print(f"Warning: Document content exceeds Milvus field limit. Truncating...")
                
                # Approach 1: Simple truncation (last resort)
                # truncated_content = content[:self.MAX_CONTENT_LENGTH]
                
                # Approach 2: Truncate at sentence/paragraph boundaries
                truncated_content = self._smart_truncate(content, self.MAX_CONTENT_LENGTH)
                
                # Create modified document
                modified_doc = Document(
                    page_content=truncated_content,
                    metadata={
                        **doc.metadata,
                        "truncated": True,
                        "original_length": len(content)
                    }
                )
                processed_documents.append(modified_doc)
                processed_embeddings.append(embeddings[i])
            else:
                # Content is within limits
                processed_documents.append(doc)
                processed_embeddings.append(embeddings[i])
        
        # Call the original add_documents with processed content
        return self.milvus_store.add_documents(processed_documents, processed_embeddings)
    
    def _smart_truncate(self, text: str, max_length: int) -> str:
        """
        Intelligently truncate text to stay under max_length while preserving
        semantic boundaries like sentences or paragraphs.
        
        Args:
            text: The text to truncate
            max_length: Maximum allowed length
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
            
        # Try truncating at paragraph boundaries first
        truncated = text[:max_length]
        last_para = truncated.rfind("\n\n")
        
        # If we found a paragraph break that's reasonably far in, use it
        if last_para > max_length * 0.8:
            return truncated[:last_para]
            
        # Try sentence boundaries next
        last_sentence = max(
            truncated.rfind(". "),
            truncated.rfind("! "),
            truncated.rfind("? ")
        )
        
        # If we found a sentence break that's reasonably far in, use it
        if last_sentence > max_length * 0.7:
            return truncated[:last_sentence+1]  # Include the period
            
        # Fall back to simple truncation with ellipsis
        return truncated[:max_length-3] + "..."
    
    def similarity_search(self, query_embedding, metadata_filter=None, top_k=3):
        """Pass through to the original similarity_search."""
        return self.milvus_store.similarity_search(
            query_embedding, metadata_filter, top_k
        )
    
    def release(self):
        """Pass through to the original release method."""
        return self.milvus_store.release()

class RAGSystem:
    """Main RAG system that combines all components"""
    
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2",
        ollama_embed_model: str = "llama3.2",
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        collection_name: str = "document_store",
        temperature: float = 0.7
    ):
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator(
            model_name=ollama_embed_model,
            base_url=ollama_base_url
        )
        self.vector_store = MilvusVectorStore(
            host=milvus_host,
            port=milvus_port,
            collection_name=collection_name,
            dim=self.embedding_generator.embedding_dim  # This might be None initially
        )
        
        # Initialize the LLM
        self.llm = OllamaLLM(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=temperature
        )
        
        # Set up RAG prompt template
        self.prompt_template = PromptTemplate(
            template="""
            Answer the question based on the context provided.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """,
            input_variables=["context", "question"]
        )
    
    def ingest_document(self, file_path: str):
        """Process and store a document in the vector database"""
        # Load document
        documents = self.document_processor.load_document(file_path)
        
        # Split into chunks
        chunks = self.document_processor.split_documents(documents)
        
        # Generate embeddings
        texts = [chunk.page_content for chunk in chunks]
        
        # Handle empty documents case
        if not texts:
            print("No text chunks extracted from document")
            return 0
            
        # For debugging
        print(f"Processing {len(texts)} text chunks")
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Log embedding dimensions
        if embeddings and len(embeddings) > 0:
            print(f"Embedding dimension: {len(embeddings[0])}")
        
        # Store in Milvus
        self.vector_store.add_documents(chunks, embeddings)
        
        return len(chunks)
    
    def query(self, question: str, top_k: int = 3):
        """Query the RAG system with a question"""
        # Generate embedding for the question
        query_embedding = self.embedding_generator.generate_embedding(question)
        
        # Retrieve relevant documents
        results = self.vector_store.similarity_search(query_embedding, top_k=top_k)
        
        # Format context from retrieved documents
        context = "\n\n".join([doc["content"] for doc in results])
        
        # Generate answer
        chain = self.prompt_template | self.llm
        answer = chain.invoke({"context": context, "question": question})
        
        return {
            "answer": answer,
            "sources": results
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.vector_store.release()


# Example usage
def main():
    # Initialize RAG system
    rag = RAGSystem(
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.2",
        milvus_host="localhost",
        milvus_port="19530"
    )
    
    # Ingest documents          
    doc_path = "Majorana1 whitepaper.pdf"
    chunks_count = rag.ingest_document(doc_path)
    print(f"Ingested document into {chunks_count} chunks")
    
    # Query the system
    question = "What is RAG and how does it work?"
    result = rag.query(question)
    
    print("\n=== Question ===")
    print(question)
    
    print("\n=== Answer ===")
    print(result["answer"])
    
    print("\n=== Sources ===")
    for i, source in enumerate(result["sources"]):
        print(f"Source {i+1} (Score: {source['score']:.4f}):")
        print(source["content"][:150] + "..." if len(source["content"]) > 150 else source["content"])
        print()
    
    # Clean up
    rag.cleanup()


if __name__ == "__main__":
    main()