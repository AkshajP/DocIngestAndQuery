// frontend/docllm/lib/document-tree-util.ts
// Enhanced version with more features

/**
 * Transforms a flat list of documents into a hierarchical tree structure
 * based solely on the case_path field, suitable for AG Grid's tree data feature.
 */
export function transformDocumentsToTree(documents: any[]): any[] {
  const result: any[] = [];
  const folderMap = new Map<string, any>();
  let hasUncategorizedDocuments = false;
  
  // Process each document
  documents.forEach(doc => {
    if (!doc.case_path) {
      // Document has no case_path, will be added to "Uncategorized" folder
      // Create "Uncategorized" folder if it doesn't exist yet
      if (!folderMap.has("Uncategorized")) {
        const uncategorizedFolder = {
          name: "Uncategorized",
          path: "Uncategorized",
          isFolder: true,
          description: "Documents without a specified path",
          status: null,
          importance: null,
          page_count: null,
          size: null,
          docId: "folder_Uncategorized"
        };
        
        folderMap.set("Uncategorized", uncategorizedFolder);
        result.push(uncategorizedFolder);
      }
      
      // Add document to "Uncategorized" folder
      result.push({
        name: doc.document_name || doc.original_filename,
        docId: doc.document_id,
        path: `Uncategorized/${doc.document_name || doc.original_filename}`,
        isFolder: false,
        description: doc.original_filename || doc.document_name,
        status: doc.status,
        importance: doc.importance || determineImportance(doc),
        page_count: doc.page_count || 0,
        size: formatSize(doc.size) || 'Unknown',
      });
      
      hasUncategorizedDocuments = true;
      return;
    }
    
    // Document has case_path, parse it
    const pathParts = doc.case_path.split('/');
    const fileName = pathParts.pop(); // Last part is the filename
    
    // Create folder hierarchy from case_path
    let currentPath = '';
    for (let i = 0; i < pathParts.length; i++) {
      const folder = pathParts[i];
      // Build path incrementally (e.g., "Bundle A" then "Bundle A/Subfolder")
      currentPath = currentPath ? `${currentPath}/${folder}` : folder;
      
      // Create folder node if it doesn't exist
      if (!folderMap.has(currentPath)) {
        const folderNode = {
          name: folder,
          path: currentPath,
          isFolder: true,
          description: `${folder}`,
          status: null,
          importance: null,
          page_count: null,
          size: null,
          docId: `folder_${currentPath.replace(/\//g, '_')}` // Generate a unique folder ID
        };
        
        folderMap.set(currentPath, folderNode);
        result.push(folderNode);
      }
    }
    
    // Add document node with its case_path
    result.push({
      name: fileName,
      docId: doc.document_id,
      path: doc.case_path,
      isFolder: false,
      description: doc.original_filename || doc.document_name || fileName,
      status: doc.status,
      importance: doc.importance || determineImportance(doc),
      page_count: doc.page_count || 0,
      size: formatSize(doc.size) || 'Unknown',
    });
  });
  
  // If we created an "Uncategorized" folder but ended up with no documents in it,
  // remove it from the result
  if (!hasUncategorizedDocuments && folderMap.has("Uncategorized")) {
    const index = result.findIndex(item => item.path === "Uncategorized" && item.isFolder);
    if (index !== -1) {
      result.splice(index, 1);
    }
  }
  
  return result;
}
/**
 * Helper function to determine document importance
 */
function determineImportance(doc: any): number {
  // Just a simple placeholder implementation
  if (doc.content_types && doc.content_types.text > 10) return 3;
  if (doc.raptor_levels && doc.raptor_levels.length > 2) return 3;
  return doc.chunks_count > 10 ? 2 : 1;
}

/**
 * Helper function to format file size
 */
function formatSize(sizeInBytes?: number): string {
  if (!sizeInBytes) return 'Unknown';
  
  const KB = 1024;
  const MB = KB * 1024;
  
  if (sizeInBytes >= MB) {
    return `${(sizeInBytes / MB).toFixed(2)} MB`;
  } else {
    return `${(sizeInBytes / KB).toFixed(2)} KB`;
  }
}