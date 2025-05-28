// frontend/docllm/components/DocumentViewerSidebar.tsx
'use client';

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { documentApi } from '@/lib/api';
import { 
  LoaderIcon, 
  XIcon,
  ZoomInIcon, 
  ZoomOutIcon,
  ChevronLeftIcon,
  ChevronRightIcon
} from 'lucide-react';
import dynamic from 'next/dynamic';

// Import the PDFPageViewer component with dynamic loading - exactly like the original
const PDFPageViewer = dynamic(() => import('@/components/PDFPageViewer'), { 
  ssr: false,
  loading: () => <div className="flex justify-center items-center h-screen">
    <LoaderIcon className="animate-spin mr-2" />
    <span>Loading PDF viewer...</span>
  </div>
});

interface DocumentViewerSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  documentId: string | null;
  pageNumber?: number;
  bbox?: number[];
  sourceIndex?: number;
}

export default function DocumentViewerSidebar({
  isOpen,
  onClose,
  documentId,
  pageNumber = 1,
  bbox,
  sourceIndex
}: DocumentViewerSidebarProps) {
  // State management - matching the original document viewer
  const [documentTitle, setDocumentTitle] = useState('Document');
  const [currentPage, setCurrentPage] = useState(pageNumber);
  const [totalPages, setTotalPages] = useState(1);
  const [zoom, setZoom] = useState(0.9);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Update currentPage when pageNumber prop changes
  useEffect(() => {
    setCurrentPage(pageNumber);
  }, [pageNumber]);

  // Fetch document details when documentId changes - exactly like the original
  useEffect(() => {
    const fetchDocumentDetails = async () => {
      if (!documentId) return;
      
      try {
        setLoading(true);
        setError(null);
        // Get document details - same API call as original
        const document = await documentApi.getDocument(documentId);
        setDocumentTitle(document.original_filename || 'Document');
      } catch (err: any) {
        console.error('Error loading document details:', err);
        setError(err?.message || 'Failed to load document details');
      } finally {
        setLoading(false);
      }
    };

    if (documentId) {
      fetchDocumentDetails();
    }
  }, [documentId]);

  // Zoom controls - exactly like the original
  const handleZoomIn = () => {
    setZoom(prevZoom => Math.min(prevZoom + 0.2, 3.0));
  };

  const handleZoomOut = () => {
    setZoom(prevZoom => Math.max(prevZoom - 0.2, 0.5));
  };

  // Page navigation - exactly like the original
  const handlePreviousPage = () => {
    if (currentPage > 1) {
      setCurrentPage(prevPage => prevPage - 1);
    }
  };

  const handleNextPage = () => {
    if (currentPage < totalPages) {
      setCurrentPage(prevPage => prevPage + 1);
    }
  };

  // Page loaded handler - exactly like the original
  const handlePageLoaded = (pageCount: number) => {
    setTotalPages(pageCount);
  };

  // Handle escape key to close sidebar
  useEffect(() => {
    const handleEscapeKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscapeKey);
    return () => {
      document.removeEventListener('keydown', handleEscapeKey);
    };
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

  return (
    <>
      {/* Overlay */}
      <div 
        className="fixed inset-0 bg-black/20 z-40 transition-opacity duration-300"
        onClick={onClose}
      />
      
      {/* Sidebar */}
      <div className={`fixed right-0 top-0 h-full w-150 bg-background border-l shadow-lg z-50 transform transition-transform duration-300 ease-in-out ${
        isOpen ? 'translate-x-0' : 'translate-x-full'
      }`}>
        {/* Header - matching the original layout */}
        <div className="border-b p-4 flex items-center justify-between bg-background">
          <div className="flex items-center flex-1 min-w-0">
            <h1 className="text-lg font-semibold truncate mr-4">
              {documentTitle}
              {typeof sourceIndex === 'number' && (
                <span className="ml-2 text-sm text-muted-foreground">
                  (Source {sourceIndex + 1})
                </span>
              )}
            </h1>
          </div>
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={onClose}
            className="flex-shrink-0"
          >
            <XIcon className="h-4 w-4" />
          </Button>
        </div>

        {/* Controls - exactly matching the original viewer controls */}
        <div className="border-b p-3 flex items-center justify-between bg-background">
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleZoomOut} disabled={zoom <= 0.5}>
              <ZoomOutIcon className="h-4 w-4" />
            </Button>
            <span className="text-sm min-w-[50px] text-center">{Math.round(zoom * 100)}%</span>
            <Button variant="outline" size="sm" onClick={handleZoomIn} disabled={zoom >= 3.0}>
              <ZoomInIcon className="h-4 w-4" />
            </Button>
          </div>
          
          <div className="flex items-center gap-2">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handlePreviousPage}
              disabled={currentPage <= 1}
            >
              <ChevronLeftIcon className="h-4 w-4" />
            </Button>
            <span className="text-sm min-w-[80px] text-center">
              Page {currentPage} of {totalPages}
            </span>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleNextPage}
              disabled={currentPage >= totalPages}
            >
              <ChevronRightIcon className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Content - exactly matching the original main content */}
        <div className="flex-1 overflow-auto bg-slate-200 relative">
          {loading && !error ? (
            <div className="flex justify-center items-center h-full">
              <LoaderIcon className="animate-spin mr-2" />
              <span>Loading document details...</span>
            </div>
          ) : error ? (
            <div className="flex justify-center items-center h-full text-destructive px-4 text-center">
              <span>{error}</span>
            </div>
          ) : documentId ? (
            <PDFPageViewer
              documentId={documentId}
              pageNumber={currentPage}
              bbox={bbox}
              zoom={zoom}
              onLoaded={handlePageLoaded}
            />
          ) : (
            <div className="flex justify-center items-center h-full text-muted-foreground">
              <span>No document selected</span>
            </div>
          )}
        </div>
      </div>
    </>
  );
}