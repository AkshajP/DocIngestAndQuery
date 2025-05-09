// frontend/docllm/lib/document-tree-util.ts

interface DocumentNode {
    id: string;
    name: string;
    description?: string;
    path?: string;
    importance?: number;
    url?: string;
    status?: string;
    size?: string;
    page_count?: number;
    created_at?: string;
    children?: DocumentNode[];
    isFolder?: boolean;
    docId?: string; // For files only
  }
  
  // This simulates grouping documents into folders
  export const transformDocumentsToTree = (documents: any[]): DocumentNode[] => {
    // Create a set of departments/categories
    const departments = ['Legal', 'Finance', 'HR', 'Operations', 'Marketing'];
    
    // Create root folders
    const rootFolders: DocumentNode[] = departments.map(dept => ({
      id: dept.toLowerCase(),
      name: dept,
      isFolder: true,
      children: []
    }));
    
    // Mock descriptions for random assignment
    const descriptions = [
      'Important contract document',
      'Contains financial information',
      'Policy documentation',
      'Agreement between parties',
      'Standard operating procedure',
      'Terms and conditions document',
      'Reference material'
    ];
    
    // Mock importance levels
    const importanceLevels = [1, 2, 3, 4, 5];
    
    // Distribute documents to folders
    documents.forEach(doc => {
      // Select a random department
      const deptIndex = Math.floor(Math.random() * departments.length);
      const dept = departments[deptIndex];
      const deptFolder = rootFolders.find(f => f.name === dept);
      
      if (deptFolder && deptFolder.children) {
        // Generate all the additional metadata
        const fileSize = Math.floor(Math.random() * 2000) + 100; // 100-2100KB
        const pageCount = doc.page_count || Math.floor(Math.random() * 50) + 1; // 1-50 pages
        const description = descriptions[Math.floor(Math.random() * descriptions.length)];
        const importance = importanceLevels[Math.floor(Math.random() * importanceLevels.length)];
        const created = doc.processing_date || new Date().toISOString();
        const path = `${dept}/${doc.original_filename}`;
        
        // Add to department folder
        deptFolder.children.push({
          id: `file_${doc.document_id}`,
          name: doc.original_filename || 'Unnamed Document',
          description,
          path,
          importance,
          url: `/api/ai/documents/${doc.document_id}/file`,
          status: doc.status || 'active',
          size: `${fileSize} KB`,
          page_count: pageCount,
          created_at: created,
          isFolder: false,
          docId: doc.document_id // Keep original document ID for selection
        });
      }
    });
    
    // Filter out empty folders and sort children alphabetically
    return rootFolders
      .filter(folder => folder.children && folder.children.length > 0)
      .map(folder => ({
        ...folder,
        children: folder.children?.sort((a, b) => a.name.localeCompare(b.name))
      }))
      .sort((a, b) => a.name.localeCompare(b.name));
  };