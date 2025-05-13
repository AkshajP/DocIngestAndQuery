/**
 * Get details about a specific document
 */
export async function getDocumentDetails(documentId: string) {
    const response = await fetch(`/api/ai/documents/${documentId}`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch document details');
    }
    
    return response.json();
  }
  
  /**
   * Get a direct URL to a specific page of a document
   */
  export function getPageUrl(documentId: string, pageNumber: number) {
    return `/api/ai/documents/${documentId}/page/${pageNumber}`;
  }