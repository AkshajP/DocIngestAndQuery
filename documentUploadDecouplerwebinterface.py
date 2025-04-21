import streamlit as st
import os
import tempfile
import logging
import io
import sys
import time
import threading
import queue
from contextlib import redirect_stdout, redirect_stderr
from documentUploadDecoupler import DocumentUploader

# Set page configuration
st.set_page_config(
    page_title="Document Processor",
    page_icon="📄",
    layout="wide"
)

# Initialize session state for logs
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

# Custom logging handler that stores logs in session state
class StreamlitLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        st.session_state.log_messages.append(log_entry)
        # Keep only the most recent 200 logs to avoid memory issues
        if len(st.session_state.log_messages) > 200:
            st.session_state.log_messages = st.session_state.log_messages[-200:]

# STDOUT & STDERR capture class with better handling
class StreamCapture(io.TextIOBase):
    def __init__(self, name="OUT"):
        self.name = name
        self.content = []
    
    def write(self, text):
        if text and text.strip():  # Skip empty lines
            timestamp = time.strftime("%H:%M:%S")
            log_line = f"[{timestamp}] {self.name}: {text.rstrip()}"
            st.session_state.log_messages.append(log_line)
            # Keep log size reasonable
            if len(st.session_state.log_messages) > 200:
                st.session_state.log_messages = st.session_state.log_messages[-200:]
            # Also return the length of text so original stdout/stderr works properly
            return len(text)
        return 0
    
    def flush(self):
        pass
    
    def isatty(self):
        return False

# Set up logging
logger = logging.getLogger()  # Root logger to capture all logs
logger.setLevel(logging.INFO)

# Remove existing handlers to avoid duplicates
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add our custom handler
handler = StreamlitLogHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', '%H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Override standard output and error globally to capture all output
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = StreamCapture("STDOUT")
sys.stderr = StreamCapture("STDERR")

# Initialize document uploader
@st.cache_resource
def get_uploader():
    logger.info("Initializing DocumentUploader")
    return DocumentUploader()

uploader = get_uploader()

# Main title
st.title("📄 Document Processor")

# Function to restore original stdout/stderr when app is done
def on_app_close():
    sys.stdout = original_stdout
    sys.stderr = original_stderr

# Register the cleanup function
st.cache_resource.clear()

# Sidebar - Document list and log display
st.sidebar.title("Processed Documents")

# Log display in sidebar
st.sidebar.subheader("Processing Logs")
log_container = st.sidebar.container(height=300, border=True)
with log_container:
    log_text = "\n".join(st.session_state.log_messages)
    st.text_area("", log_text, height=270, key="log_display", disabled=True)

# Log actions button
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Clear Logs"):
        st.session_state.log_messages = []
        st.rerun()
with col2:
    if st.button("🔄 Refresh List"):
        logger.info("Refreshing document list")
        st.sidebar.success("List refreshed!")

# Get list of documents
documents = uploader.list_documents()

if not documents:
    st.sidebar.info("No documents processed yet")
else:
    # Display documents in sidebar
    for doc in documents:
        with st.sidebar.container(border=True):
            col1, col2 = st.sidebar.columns([5, 1])
            
            # Document info
            with col1:
                status_icon = "✅" if doc["status"] == "processed" else "❌"
                st.write(f"{status_icon} **{doc['filename']}**")
                
                # Display document ID and processing date
                doc_id = doc['document_id']
                # Truncate long document IDs
                # if len(doc_id) > 20:
                #     display_id = doc_id[:17] + "..."
                # else:
                display_id = doc_id
                    
                # Format date (extract just the date portion)
                proc_date = doc.get('processing_date', '')
                if proc_date and len(proc_date) >= 10:
                    display_date = proc_date[:10]  # YYYY-MM-DD format
                else:
                    display_date = "Unknown date"
                
                st.caption(f"ID: {display_id}")
                st.caption(f"Date: {display_date}")
                st.caption(f"Chunks: {doc['chunks_count']}")
            
            # Delete button
            with col2:
                if st.button("🗑️", key=f"del_{doc['document_id']}"):
                    if uploader.delete_document(doc['document_id']):
                        st.sidebar.success(f"Document deleted")
                        st.rerun()
                    else:
                        st.sidebar.error("Failed to delete")

# Main area - Document upload
uploaded_files = st.file_uploader(
    "Upload PDF documents for processing", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process Documents", type="primary"):
        # Progress tracking
        progress = st.progress(0)
        status = st.empty()
        results = st.container()
        
        # Process each file
        for i, file in enumerate(uploaded_files):
            # Status update
            status_msg = f"Processing {file.name}... ({i+1}/{len(uploaded_files)})"
            status.info(status_msg)
            logger.info(status_msg)
            
            # Save to temporary file with original filename
            original_filename = file.name
            # Create proper temp directory
            temp_dir = tempfile.mkdtemp()
            try:
                # Use original filename for temp file (fixes the name issue)
                temp_path = os.path.join(temp_dir, original_filename)
                with open(temp_path, 'wb') as tmp:
                    tmp.write(file.getbuffer())
                
                # Update user on what's happening
                logger.info(f"Saved temporary file: {temp_path}")
                
                # Fix metadata to match what DocumentUploader expects
                metadata = {
                    "original_filename": original_filename,
                    "upload_source": "streamlit_app"
                }
                
                # Create a consistent document_id based on original filename
                timestamp = int(time.time())
                safe_filename = os.path.splitext(original_filename)[0].replace(' ', '_')
                # Keep only alphanumeric chars and underscores to avoid path issues
                safe_filename = ''.join(c for c in safe_filename if c.isalnum() or c == '_')
                doc_id = f"doc_{timestamp}_{safe_filename}"
                
                # Print debug info about the document processing
                print(f"Processing document:")
                print(f"  Original filename: {original_filename}")
                print(f"  Document ID: {doc_id}")
                print(f"  Temp path: {temp_path}")
                
                # Process document with preserved filename
                logger.info(f"Starting document processing for {original_filename}")
                
                # Print a separator line to make the logs more readable
                print("-" * 60)
                print(f"PROCESSING: {original_filename}")
                print("-" * 60)
                
                result = uploader.upload_document(
                    temp_path, 
                    document_id=doc_id, 
                    metadata=metadata
                )
                
                # Print completion separator
                print("-" * 60)
                print(f"COMPLETED: {original_filename}")
                print("-" * 60)
                
                # Display result
                with results:
                    if result["status"] == "success":
                        success_msg = f"✅ {file.name} processed successfully"
                        st.success(success_msg)
                        logger.info(success_msg)
                        with st.expander("Details"):
                            st.write(f"**Document ID:** {result['document_id']}")
                            st.write(f"**Processing time:** {result['processing_time']:.2f} seconds")
                            st.write(f"**Chunks:** {result['chunks_count']}")
                            st.write(f"**RAPTOR levels:** {', '.join(map(str, result['raptor_levels']))}")
                    else:
                        error_msg = f"❌ Error processing {file.name}: {result.get('error', 'Unknown error')}"
                        st.error(error_msg)
                        logger.error(error_msg)
            
            finally:
                # Clean up temporary files
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        logger.info(f"Removed temporary file: {temp_path}")
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
                        logger.info(f"Removed temporary directory: {temp_dir}")
                except Exception as e:
                    logger.error(f"Error cleaning up temporary files: {str(e)}")
                
                # Update progress
                progress.progress((i + 1) / len(uploaded_files))
        
        # Final status
        completion_msg = "All documents processed!"
        status.success(completion_msg)
        logger.info(completion_msg)
        
        # Refresh button
        if st.button("Refresh Document List"):
            st.rerun()

# Document details section
st.divider()
st.header("Document Details")

# Document selector
if documents:
    selected_doc = st.selectbox(
        "Select a document to view details",
        options=[doc["document_id"] for doc in documents],
        format_func=lambda x: next((doc["filename"] for doc in documents if doc["document_id"] == x), x)
    )
    
    if selected_doc:
        metadata = uploader.get_document_metadata(selected_doc)
        
        if metadata:
            # Display document info
            st.subheader(f"📄 {metadata.get('original_filename', 'Unknown')}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chunks", metadata.get('chunks_count', 0))
            with col2:
                st.metric("Processing Time", f"{metadata.get('processing_time', 0):.2f}s")
            with col3:
                st.metric("Status", metadata.get('status', 'Unknown'))
            
            # Content types
            if "content_types" in metadata and metadata["content_types"]:
                st.subheader("Content Types")
                for content_type, count in metadata["content_types"].items():
                    st.write(f"- **{content_type}**: {count}")
            
            # RAPTOR levels
            if "raptor_levels" in metadata and metadata["raptor_levels"]:
                st.subheader("RAPTOR Levels")
                st.write(", ".join([f"Level {level}" for level in sorted(metadata["raptor_levels"])]))
        else:
            st.error("Could not load document metadata")
else:
    st.info("No documents available. Upload and process documents to view details.")