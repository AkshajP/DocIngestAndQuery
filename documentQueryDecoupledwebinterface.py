import streamlit as st
import os
import time
import logging
import json
from documentQueryDecoupled import DiskBasedDocumentQuerier, format_document_table, format_history_display

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('query_app')

# Page config
st.set_page_config(
    page_title="Document Query Chatbot",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your document assistant. Load documents from the sidebar and ask me questions about them."}
    ]
if "loaded_documents" not in st.session_state:
    st.session_state.loaded_documents = []
if "use_tree" not in st.session_state:
    st.session_state.use_tree = False
if "use_history" not in st.session_state:
    st.session_state.use_history = True
if "querier" not in st.session_state:
    # Initialize the document querier - this ensures the same instance is used throughout the session
    st.session_state.querier = DiskBasedDocumentQuerier(max_history_length=10)
    # Add initial greeting to chat history
    st.session_state.querier.chat_history.add_interaction(
        "initial_greeting", 
        "Hello! I'm your document assistant. Load documents from the sidebar and ask me questions about them."
    )
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# Main title
st.title("📚 Document Query Assistant")

# Sidebar - Document Management
st.sidebar.title("Document Management")

# Refresh button for document list
if st.sidebar.button("🔄 Refresh Document List"):
    st.session_state.querier.refresh_registry()
    st.sidebar.success("Document list refreshed!")
    st.rerun()

# Get available documents
documents = st.session_state.querier.list_available_documents()

# Document selection section
st.sidebar.subheader("Available Documents")
if not documents:
    st.sidebar.info("No documents available in storage.")
else:
    # Display available documents with checkboxes
    selected_docs = []
    for doc in documents:
        doc_id = doc.get("document_id", "")
        filename = doc.get("filename", "Unknown")
        is_loaded = doc_id in st.session_state.loaded_documents
        
        # Create a checkbox for each document
        selected = st.sidebar.checkbox(
            f"{filename}", 
            value=is_loaded,
            key=f"check_{doc_id}"
        )
        
        if selected:
            selected_docs.append(doc_id)
    
    # Update loaded documents based on selection
    for doc_id in selected_docs:
        if doc_id not in st.session_state.loaded_documents:
            # Load new document
            with st.spinner(f"Loading {doc_id}..."):
                result = st.session_state.querier.load_document(doc_id)
                if result["status"] == "success":
                    st.session_state.loaded_documents.append(doc_id)
                    st.sidebar.success(f"Loaded: {doc_id}")
                else:
                    st.sidebar.error(f"Failed to load: {doc_id}")
    
    # Check for documents to unload
    for doc_id in list(st.session_state.loaded_documents):
        if doc_id not in selected_docs:
            # Unload document
            with st.sidebar.spinner(f"Unloading {doc_id}..."):
                result = st.session_state.querier.unload_document(doc_id)
                if result["status"] == "success":
                    st.session_state.loaded_documents.remove(doc_id)
                    st.sidebar.success(f"Unloaded: {doc_id}")
                else:
                    st.sidebar.error(f"Failed to unload: {doc_id}")

# Display currently loaded documents
st.sidebar.subheader("Currently Loaded Documents")
if not st.session_state.loaded_documents:
    st.sidebar.info("No documents currently loaded")
else:
    for doc_id in st.session_state.loaded_documents:
        # Find document details
        doc_info = next((doc for doc in documents if doc.get("document_id") == doc_id), None)
        if doc_info:
            filename = doc_info.get("filename", "Unknown")
            chunks = doc_info.get("chunks_count", 0)
            st.sidebar.success(f"📄 {filename} ({chunks} chunks)")
        else:
            st.sidebar.success(f"📄 {doc_id}")

# Query settings
st.sidebar.subheader("Query Settings")

# Toggle for tree vs flat traversal
st.session_state.use_tree = st.sidebar.toggle(
    "Use Tree-based Retrieval", 
    value=st.session_state.use_tree,
    help="Toggle between flat (off) and tree-based (on) retrieval"
)

# Toggle for using history
st.session_state.use_history = st.sidebar.toggle(
    "Use Chat History", 
    value=st.session_state.use_history,
    help="Use conversation history for context in responses"
)

# Toggle for showing sources
st.session_state.show_sources = st.sidebar.toggle(
    "Show Sources", 
    value=st.session_state.show_sources,
    help="Show the document sources used for responses"
)

# Slider for top_k
st.session_state.top_k = st.sidebar.slider(
    "Number of chunks to retrieve", 
    min_value=1, 
    max_value=10, 
    value=st.session_state.top_k,
    help="Higher values may give more comprehensive but slower responses"
)

# Chat history management
st.sidebar.subheader("Chat History")

# Add debug mode toggle
st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", value=st.session_state.debug_mode)

if st.sidebar.button("Clear Chat History"):
    # This ensures the chat history is properly cleared in the DiskBasedDocumentQuerier instance
    st.session_state.querier.clear_history()
    # Reset messages but keep the welcome message
    st.session_state.messages = [
        {"role": "assistant", "content": "Chat history cleared. You can start a new conversation."}
    ]
    # Add initial message to history after clearing
    st.session_state.querier.chat_history.add_interaction(
        "system_message", 
        "Chat history cleared. You can start a new conversation."
    )
    st.sidebar.success("Chat history cleared")

# Display chat history
with st.sidebar.expander("View Chat History", expanded=True):
    history = st.session_state.querier.get_chat_history()
    if not history:
        st.write("No chat history available")
    else:
        # Display raw history in debug mode
        if st.session_state.debug_mode:
            st.subheader("Debug: Raw History Data")
            st.code(json.dumps(history, indent=2), language="json")
            
            # Show the formatted history as seen by the LLM
            st.subheader("Formatted History (as seen by the LLM)")
            formatted_history = st.session_state.querier.chat_history.get_formatted_history()
            st.code(formatted_history, language="markdown")
        
        # Always show the history in a readable format
        st.subheader("Conversation History")
        for i, item in enumerate(history):
            if "type" in item and item["type"] == "summary":
                st.write(f"**Summary of {item['original_count']} interactions:**")
                st.write(item["content"])
                st.write("---")
            else:
                st.write(f"**Q{i+1}:** {item.get('question', '')}")
                answer = item.get('answer', '')
                if len(answer) > 100:
                    st.write(f"**A{i+1}:** {answer[:100]}...")
                else:
                    st.write(f"**A{i+1}:** {answer}")
                st.write("---")

# Main chat interface
st.divider()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if enabled and available for assistant messages
        if message["role"] == "assistant" and st.session_state.show_sources and "sources" in message and message["sources"]:
            with st.expander("View Sources", expanded=False):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}** (Score: {source['score']:.4f})")
                    st.markdown(f"Document: {source['document_id']}")
                    
                    # Show content with proper truncation
                    content = source.get('content', '')
                    if len(content) > 300:
                        st.markdown(f"{content[:300]}...")
                    else:
                        st.markdown(content)
                    
                    # Show metadata if available
                    if "metadata" in source and source["metadata"]:
                        metadata = source["metadata"]
                        if "page_idx" in metadata:
                            st.markdown(f"Page: {metadata['page_idx']}")
                        if "type" in metadata:
                            st.markdown(f"Type: {metadata['type']}")
                    
                    st.markdown("---")

# Input for new question
if st.session_state.loaded_documents:
    query = st.chat_input("Ask a question about the loaded documents...")
    if query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Debug mode - display message about query processing
        if st.session_state.debug_mode:
            with st.expander("Debug: Processing Query", expanded=True):
                st.info(f"Processing query: '{query}'")
                st.info(f"Use history: {st.session_state.use_history}")
                st.info(f"Use tree: {st.session_state.use_tree}")
                st.info(f"Top K: {st.session_state.top_k}")
                
                # Show current history before adding this query
                st.subheader("Current Chat History")
                history = st.session_state.querier.get_chat_history()
                if history:
                    st.code(json.dumps(history, indent=2), language="json")
        
        # Generate response
        with st.chat_message("assistant"):
            # Process query with error handling
            message_placeholder = st.empty()
            sources_container = st.container()
            time_placeholder = st.empty()
            
            try:
                with st.spinner("Thinking..."):
                    start_time = time.time()
                    # This is where the query is processed, ensuring chat history is used
                    result = st.session_state.querier.query(
                        question=query,
                        document_ids=st.session_state.loaded_documents,
                        top_k=st.session_state.top_k,
                        use_tree=st.session_state.use_tree,
                        use_history=st.session_state.use_history
                    )
                    
                    # Display answer
                    message_placeholder.markdown(result["answer"])
                    
                    # Show sources if enabled
                    if st.session_state.show_sources and result["sources"]:
                        with sources_container.expander("View Sources", expanded=False):
                            for i, source in enumerate(result["sources"]):
                                st.markdown(f"**Source {i+1}** (Score: {source['score']:.4f})")
                                st.markdown(f"Document: {source['document_id']}")
                                
                                # Show content with proper truncation
                                content = source.get('content', '')
                                if len(content) > 300:
                                    st.markdown(f"{content[:300]}...")
                                else:
                                    st.markdown(content)
                                
                                # Show metadata if available
                                if "metadata" in source and source["metadata"]:
                                    metadata = source["metadata"]
                                    if "page_idx" in metadata:
                                        st.markdown(f"Page: {metadata['page_idx']}")
                                    if "type" in metadata:
                                        st.markdown(f"Type: {metadata['type']}")
                                
                                st.markdown("---")
                    
                    # Display query time
                    time_taken = time.time() - start_time
                    retrieval_method = "tree-based" if st.session_state.use_tree else "flat"
                    time_placeholder.caption(f"Query processed in {time_taken:.2f} seconds using {retrieval_method} retrieval")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result["answer"],
                        "sources": result["sources"]
                    })
                    
                    # In debug mode, show what was added to history
                    if st.session_state.debug_mode:
                        with st.expander("Debug: Chat History Update", expanded=True):
                            st.info("The above Q&A has been added to the chat history.")
                            history = st.session_state.querier.get_chat_history()
                            if history:
                                st.code(json.dumps(history[-1], indent=2), language="json")
            
            except Exception as e:
                error_message = f"Error processing query: {str(e)}"
                message_placeholder.error(error_message)
                logger.error(error_message)
                
                # Add error message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"⚠️ {error_message}"
                })
else:
    # Display message if no documents are loaded
    st.info("👈 Please load at least one document from the sidebar to start chatting.")

# Display app info at the bottom
with st.expander("About This App", expanded=False):
    st.markdown("""
    This is a document query interface that uses RAG (Retrieval-Augmented Generation) to answer questions about loaded documents.
        
    **Features:**
    - Load multiple documents from the document store
    - Toggle between flat and tree-based retrieval methods
    - Use chat history for context in responses
    - View the sources of information used for answers
    
    **Settings:**
    - **Tree-based Retrieval**: Uses hierarchical document structure for more efficient retrieval
    - **Chat History**: Uses previous conversation for context
    - **Show Sources**: Displays the document sources used for answers
    - **Number of chunks**: Controls how many document chunks are retrieved for each query
    """)