// lib/document-tree-util.js
/**
 * Transforms a flat list of documents into a hierarchical tree structure for AG Grid
 * using the case_path field when available.
 * 
 * @param {Array} documents - The list of documents from the API
 * @returns {Array} - Tree-structured data for AG Grid
 */
export function transformDocumentsToTree(documents) {
  console.log(documents)
  const treeData = [];
  
  // Track folders to avoid duplication
  const folderSet = new Set();
  
  // First, add all folders from case_paths
  documents.forEach(doc => {
    if (doc.case_path) {
      // Extract folder name from case_path
      const pathParts = doc.case_path.split('/');
      const folderName = pathParts[0];
      
      // Add folder node if it doesn't exist yet
      if (!folderSet.has(folderName)) {
        folderSet.add(folderName);
        treeData.push({
          isFolder: true,
          name: folderName,
          docId: `folder_${folderName}`, // Add ID for the folder for selection purposes
          path: folderName, // Path for the folder is just its name
        });
      }
    }
  });
  
  // Then add all documents
  documents.forEach(doc => {
    // Format file size (if available)
    const size = formatFileSize(doc.file_size);
    
    // Basic document properties
    const docData = {
      docId: doc.document_id,
      name: doc.original_filename || 'Unnamed Document',
      status: doc.status || 'unknown',
      page_count: doc.page_count || 0,
      size: size,
      importance: calculateImportance(doc), // Function to determine importance
      description: generateDescription(doc), // Function to generate description
    };
    
    // If document has case_path, use it for hierarchical display
    if (doc.case_path) {
      docData.path = doc.case_path;
    } else {
      // For documents without case_path, set a flag to show at root level
      docData.path = 'Uncategorized/' + doc.original_filename;
    }
    
    treeData.push(docData);
  });
  
  // Add uncategorized folder if there are documents without case_path
  if (documents.some(doc => !doc.case_path)) {
    treeData.unshift({
      isFolder: true,
      name: 'Uncategorized',
      docId: 'folder_uncategorized',
      path: 'Uncategorized',
    });
  }
  
  return treeData;
}

/**
 * Formats file size in B, KB, or MB
 */
function formatFileSize(bytes) {
  if (!bytes) return 'Unknown';
  
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Calculate importance rating (1-5) based on document properties
 */
function calculateImportance(doc) {
  // Sample logic to determine document importance
  // Could be based on file size, page count, content types, etc.
  if (!doc) return 1;
  
  let importance = 3; // Default
  
  // Factors that could increase importance
  if (doc.page_count > 20) importance++;
  if (doc.chunks_count > 50) importance++;
  
  // Factors that could decrease importance
  if (doc.status !== 'processed') importance--;
  
  // Ensure importance is within range 1-5
  return Math.max(1, Math.min(5, importance));
}

/**
 * Generate a meaningful description for the document
 */
function generateDescription(doc) {
  if (!doc) return '';
  
  let description = '';
  
  if (doc.status === 'processed') {
    description += `${doc.chunks_count || 0} chunks`;
    
    if (doc.content_types) {
      const contentTypes = [];
      if (doc.content_types.text) contentTypes.push(`${doc.content_types.text} text sections`);
      if (doc.content_types.table) contentTypes.push(`${doc.content_types.table} tables`);
      if (doc.content_types.image) contentTypes.push(`${doc.content_types.image} images`);
      
      if (contentTypes.length > 0) {
        description += ` (${contentTypes.join(', ')})`;
      }
    }
  } else if (doc.status === 'processing') {
    description = 'Currently processing...';
  } else if (doc.status === 'failed') {
    description = 'Processing failed';
  } else if (doc.status === 'pending') {
    description = 'Waiting to be processed';
  } else {
    description = 'Status unknown';
  }
  
  return description;
}