// frontend/docllm/components/admin/DocumentViewerModal.tsx
'use client';

import { useEffect, useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { adminApi } from '@/lib/api';
import PDFPageViewer from '@/components/PDFPageViewer';
import { ChevronLeft, ChevronRight, X, Download } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface DocumentViewerModalProps {
  isOpen: boolean;
  onClose: () => void;
  documentId: string;
  documentName: string;
}

export default function DocumentViewerModal({
  isOpen,
  onClose,
  documentId,
  documentName
}: DocumentViewerModalProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chunks, setChunks] = useState<any[]>([]);
  const [selectedChunkIndex, setSelectedChunkIndex] = useState(0);
  const [currentPage, setCurrentPage] = useState(1); // UI uses 1-based page numbers
  const [totalPages, setTotalPages] = useState(0);
  const [pageChunks, setPageChunks] = useState<any[]>([]);
  const [zoom, setZoom] = useState(1.0);

  // Load chunks when the modal opens
  useEffect(() => {
    if (isOpen && documentId) {
      loadChunks();
    }
  }, [isOpen, documentId]);

  // Update page chunks when current page or chunks change
  useEffect(() => {
    if (chunks.length > 0) {
      // Filter chunks for the current page - DB uses 0-based page numbers
      const chunksOnPage = chunks.filter(chunk => chunk.page_number === currentPage - 1);
      setPageChunks(chunksOnPage);
      
      // Reset selection to the first chunk if available
      if (chunksOnPage.length > 0) {
        setSelectedChunkIndex(0);
      } else {
        setSelectedChunkIndex(-1);
      }
    } else {
      setPageChunks([]);
      setSelectedChunkIndex(-1);
    }
  }, [chunks, currentPage]);

const getChunkColor = (index: number): string => {
  // Array of distinct colors - you can expand this with more colors
  const colors = [
    '#FF5733', // Red-orange
    '#33FF57', // Green
    '#3357FF', // Blue
    '#FF33F5', // Pink
    '#33FFF5', // Cyan
    '#F5FF33', // Yellow
    '#FF8333', // Orange
    '#8333FF', // Purple
    '#33FFBD', // Mint
    '#FF3355', // Red
    '#33C4FF', // Light blue
    '#FF33BD', // Magenta
    '#C4FF33', // Lime
    '#FF5733', // Coral
    '#33FF8A', // Seafoam
    '#8A33FF'  // Lavender
  ];
  
  // Get color based on index, cycling through if needed
  return colors[index % colors.length];
};
  const loadChunks = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await adminApi.getDocumentChunks(documentId, { chunkType: 'original' });
      
      if (response && response.chunks) {
        setChunks(response.chunks);
        
        // Determine the initial page to show
        if (response.chunks.length > 0) {
          // Find the first chunk with valid page number and add 1 (convert to 1-based)
          const firstChunk = response.chunks.find(chunk => 
            chunk.page_number !== undefined && chunk.page_number !== null
          );
          
          if (firstChunk) {
            // Convert 0-based DB page to 1-based UI page
            setCurrentPage((firstChunk.page_number || 0) + 1);
          } else {
            setCurrentPage(1); // Default to first page
          }
        }
      } else {
        setError('No chunks found for this document');
      }
    } catch (err: any) {
      console.error('Error loading document chunks:', err);
      setError(err?.message || 'Failed to load document chunks');
    } finally {
      setLoading(false);
    }
  };

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
      // Selected chunk index will be reset in the useEffect
    }
  };

  const handleChunkSelect = (index: number) => {
    if (index >= 0 && index < pageChunks.length) {
      setSelectedChunkIndex(index);
    }
  };

  const getSelectedBoundingBox = () => {
    if (pageChunks.length > 0 && selectedChunkIndex >= 0 && selectedChunkIndex < pageChunks.length) {
      const chunk = pageChunks[selectedChunkIndex];
      if (chunk.bounding_boxes && chunk.bounding_boxes.length > 0) {
        return chunk.bounding_boxes[0]; // Return the first bounding box
      }
      
      // Fallback to bbox in metadata if available
      if (chunk.metadata && chunk.metadata.bbox) {
        return chunk.metadata.bbox;
      }
    }
    return undefined;
  };

  const getAllPageBoundingBoxes = () => {
  return pageChunks
    .filter(chunk => 
      (chunk.bounding_boxes && chunk.bounding_boxes.length > 0) ||
      (chunk.metadata && chunk.metadata.bbox)
    )
    .map((chunk, index) => {
      // Get the bbox either from bounding_boxes or metadata.bbox
      let bboxData = chunk.bounding_boxes && chunk.bounding_boxes.length > 0 
        ? chunk.bounding_boxes[0] 
        : chunk.metadata?.bbox;
      
      // Extract the actual coordinates - this is the key fix
      // If bboxData is an object with a bbox property, extract that array
      const coordinates = bboxData && typeof bboxData === 'object' && 'bbox' in bboxData 
        ? bboxData.bbox  // Extract the nested bbox array
        : bboxData;      // Use directly if it's already an array
        
      return {
        bbox: coordinates, // This should now be a direct array of 4 numbers
        isActive: index === selectedChunkIndex,
        color: getChunkColor(index)
      };
    });
};

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-screen-xl w-[90vw] max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader className="flex flex-row items-center justify-between">
          <DialogTitle>
            Document Viewer: {documentName || documentId}
          </DialogTitle>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setZoom(prev => Math.max(0.5, prev - 0.25))}
            >
              -
            </Button>
            <span className="mx-2">{zoom.toFixed(1)}x</span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setZoom(prev => Math.min(3, prev + 0.25))}
            >
              +
            </Button>
             {/* <Button
              variant="outline"
              size="sm"
              asChild
            >
              <a href={`/api/ai/documents/${documentId}/file`} download target="_blank">
                <Download className="h-4 w-4 mr-1" />
                Download
              </a>
            </Button> */}
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
            >
              {/* <X className="h-4 w-4" /> */}
            </Button> 
          </div>
        </DialogHeader>
        
        {loading ? (
          <div className="flex items-center justify-center h-[600px]">
            <div className="animate-spin h-8 w-8 border-t-2 border-b-2 border-primary rounded-full"></div>
            <span className="ml-2">Loading document chunks...</span>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-[600px] text-destructive">
            {error}
          </div>
        ) : (
          <>
            <div className="flex flex-col md:flex-row h-full gap-4">
              {/* PDF Viewer (Left Side) */}
              <div className="flex-1 overflow-hidden border rounded-md">
                <PDFPageViewer
                documentId={documentId}
                pageNumber={currentPage}
                bbox={getSelectedBoundingBox()}
                zoom={zoom}
                onLoaded={(pageCount) => setTotalPages(pageCount)}
                allBoundingBoxes={getAllPageBoundingBoxes()}
                />
              </div>
              
              {/* All Chunks for Current Page (Right Side) */}
              <div className="flex-1 flex flex-col border rounded-md overflow-hidden">
  <div className="p-4 border-b bg-muted/30 sticky top-0 z-10">
    {/* Header content remains the same */}
    <div className="flex items-center justify-between mb-2">
      <h3 className="font-medium">
        {pageChunks.length} Chunks on Page {currentPage}
      </h3>
    </div>
    
    <div className="flex gap-2 mb-2">
      <Button
        variant="outline"
        size="sm"
        onClick={() => handlePageChange(currentPage - 1)}
        disabled={currentPage <= 1}
      >
        <ChevronLeft className="h-4 w-4 mr-1" />
        Previous Page
      </Button>
      <span className="flex items-center px-2">
        Page {currentPage} of {totalPages}
      </span>
      <Button
        variant="outline"
        size="sm"
        onClick={() => handlePageChange(currentPage + 1)}
        disabled={currentPage >= totalPages}
      >
        Next Page
        <ChevronRight className="h-4 w-4 ml-1" />
      </Button>
    </div>
  </div>
  
  {/* Update the scrollable container */}
  <div className="flex-1 overflow-y-auto" style={{ maxHeight: "calc(80vh - 120px)" }}>
    {pageChunks.length > 0 ? (
      <div className="divide-y">
        {pageChunks.map((chunk, index) => {
          const chunkColor = getChunkColor(index);
          return (
            <div 
              key={`chunk-${index}-${chunk.chunk_id || 'unknown'}`}
              className={`p-4 ${selectedChunkIndex === index ? 'bg-muted/30' : ''} cursor-pointer hover:bg-muted/20`}
              onClick={() => handleChunkSelect(index)}
            >
              <div className="mb-2 text-xs font-mono flex justify-between items-center">
                <div className="flex items-center">
                  {/* Color indicator matching the bbox */}
                  <span 
                    className="inline-block h-4 w-4 mr-2 rounded-sm" 
                    style={{ backgroundColor: chunkColor, border: '1px solid rgba(0,0,0,0.1)' }}
                  ></span>
                  <strong>Chunk {index + 1}</strong>
                </div>
                {chunk.bounding_boxes && (
                  <span className="text-muted-foreground">
                    {JSON.stringify(chunk.bounding_boxes[0]).slice(0, 30)}...
                  </span>
                )}
              </div>
              
              {/* Wrap the content with a border of the same color */}
             <div 
                className="p-3 rounded-md bg-white text-sm overflow-auto"
                style={{ border: `2px solid ${chunkColor}` }}
                >
                <ReactMarkdown 
                    remarkPlugins={[remarkGfm]} 
                >
                    {chunk.content}
                </ReactMarkdown>
                </div>
            </div>
          );
        })}
      </div>
    ) : (
      <div className="flex items-center justify-center h-full text-muted-foreground p-4">
        No chunks available for this page
      </div>
    )}
  </div>
</div></div>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}