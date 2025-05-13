// frontend/docllm/app/(main)/documents/viewer/[id]/page.tsx
'use client';

import { useEffect, useState } from 'react';
import { useParams, useSearchParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { documentApi } from '@/lib/api';
import { 
  LoaderIcon, 
  ArrowLeftIcon, 
  ZoomInIcon, 
  ZoomOutIcon,
  ChevronLeftIcon,
  ChevronRightIcon
} from 'lucide-react';
import dynamic from 'next/dynamic';

// Import the PDFPageViewer component with dynamic loading
const PDFPageViewer = dynamic(() => import('@/components/PDFPageViewer'), { 
  ssr: false,
  loading: () => <div className="flex justify-center items-center h-screen">
    <LoaderIcon className="animate-spin mr-2" />
    <span>Loading PDF viewer...</span>
  </div>
});

export default function DocumentViewerPage() {
  const { id } = useParams();
  const searchParams = useSearchParams();
  const documentId = Array.isArray(id) ? id[0] : id;
  
  // Parse query parameters
  const pageParam = searchParams.get('page');
  const bboxParam = searchParams.get('bbox');
  const sourceIndexParam = searchParams.get('sourceIndex');
  
  // Parse page number (1-based in URL, but we need to adjust for display)
  const initialPage = pageParam ? parseInt(pageParam, 10) : 1;
  
  // Parse bbox if available
  const bbox = bboxParam ? bboxParam.split(',').map(Number) : undefined;
  
  // State
  const [documentTitle, setDocumentTitle] = useState('Document');
  const [currentPage, setCurrentPage] = useState(initialPage);
  const [totalPages, setTotalPages] = useState(1);
  const [zoom, setZoom] = useState(1.2);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDocumentDetails = async () => {
      try {
        setLoading(true);
        // Get document details
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

  const handleZoomIn = () => {
    setZoom(prevZoom => Math.min(prevZoom + 0.2, 3.0));
  };

  const handleZoomOut = () => {
    setZoom(prevZoom => Math.max(prevZoom - 0.2, 0.5));
  };

  const handlePreviousPage = () => {
    if (currentPage > 1) {
      setCurrentPage(prevPage => prevPage - 1);
      // Update URL to reflect the new page
      const url = new URL(window.location.href);
      url.searchParams.set('page', (currentPage - 1).toString());
      window.history.pushState({}, '', url);
    }
  };

  const handleNextPage = () => {
    if (currentPage < totalPages) {
      setCurrentPage(prevPage => prevPage + 1);
      // Update URL to reflect the new page
      const url = new URL(window.location.href);
      url.searchParams.set('page', (currentPage + 1).toString());
      window.history.pushState({}, '', url);
    }
  };

  const handlePageLoaded = (pageCount: number) => {
    setTotalPages(pageCount);
  };

  const handleGoBack = () => {
    window.history.back();
  };

  return (
    <div className="flex flex-col h-screen">
      <header className="border-b p-4 flex items-center justify-between bg-background">
        <div className="flex items-center">
          <Button variant="ghost" onClick={handleGoBack}>
            <ArrowLeftIcon className="mr-2 h-4 w-4" />
            Back
          </Button>
          <h1 className="ml-4 text-xl font-semibold truncate max-w-md">
            {documentTitle}
            {sourceIndexParam && <span className="ml-2 text-sm text-muted-foreground">(Source {parseInt(sourceIndexParam, 10) + 1})</span>}
          </h1>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleZoomOut} disabled={zoom <= 0.5}>
            <ZoomOutIcon className="h-4 w-4" />
          </Button>
          <span className="mx-2">{Math.round(zoom * 100)}%</span>
          <Button variant="outline" size="sm" onClick={handleZoomIn} disabled={zoom >= 3.0}>
            <ZoomInIcon className="h-4 w-4" />
          </Button>
          
          <div className="flex items-center ml-4">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handlePreviousPage}
              disabled={currentPage <= 1}
            >
              <ChevronLeftIcon className="h-4 w-4" />
            </Button>
            <span className="mx-2">
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
      </header>

      <main className="flex-grow relative overflow-auto bg-slate-200">
        {loading && !error ? (
          <div className="flex justify-center items-center h-full">
            <LoaderIcon className="animate-spin mr-2" />
            <span>Loading document details...</span>
          </div>
        ) : error ? (
          <div className="flex justify-center items-center h-full text-destructive">
            <span>{error}</span>
          </div>
        ) : (
          <PDFPageViewer
            documentId={documentId}
            pageNumber={currentPage}
            bbox={bbox}
            zoom={zoom}
            onLoaded={handlePageLoaded}
          />
        )}
      </main>
    </div>
  );
}