// frontend/docllm/components/PDFPageViewer.tsx
'use client';

import { useEffect, useRef, useState } from 'react';
import * as pdfjs from 'pdfjs-dist';
import { PDFDocumentProxy, PDFPageProxy } from 'pdfjs-dist';
import { LoaderIcon } from 'lucide-react';

// Initialize PDF.js worker only on client
if (typeof window !== 'undefined' && !pdfjs.GlobalWorkerOptions.workerSrc) {
    pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;
  }

interface PDFPageViewerProps {
  documentId: string;
  pageNumber: number;
  bbox?: number[];
  zoom?: number;
  onLoaded?: (pageCount: number) => void;
}

export default function PDFPageViewer({
  documentId,
  pageNumber,
  bbox,
  zoom = 1.0,
  onLoaded
}: PDFPageViewerProps) {
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
  const [page, setPage] = useState<PDFPageProxy | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const annotationLayerRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Load only the specific PDF page
  useEffect(() => {
    const loadPdfPage = async () => {
      try {
        setLoading(true);
        
        // Create a URL for the specific page
        //http://localhost:3000/api/ai/documents/doc_1745913377_output/view
        const pdfUrl = `/api/ai/documents/${documentId}/view`;
        
        // Load the document (but we'll only render the requested page)
        const loadingTask = pdfjs.getDocument(pdfUrl);
        const document = await loadingTask.promise;
        
        setPdf(document);
        onLoaded?.(document.numPages);
        
        // Load the specific page
        const pdfPage = await document.getPage(pageNumber);
        setPage(pdfPage);
        
        // Render the page
        renderPage(pdfPage, zoom);
        
        setLoading(false);
      } catch (err: any) {
        console.error('Error loading PDF page:', err);
        setError(err?.message || 'Failed to load PDF page');
        setLoading(false);
      }
    };

    loadPdfPage();

    // Cleanup
    return () => {
      if (page) {
        page.cleanup();
      }
    };
  }, [documentId, pageNumber]);

  // Re-render when zoom changes
  useEffect(() => {
    if (page) {
      renderPage(page, zoom);
    }
  }, [zoom, page]);

  // Render the PDF page
  const renderPage = async (pdfPage: PDFPageProxy, scale: number) => {
    if (!canvasRef.current) return;

    try {
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');

      if (!context) {
        throw new Error('Canvas context is null');
      }

      // Calculate viewport with zoom
      const viewport = pdfPage.getViewport({ scale: scale });

      // Set canvas dimensions
      canvas.height = viewport.height;
      canvas.width = viewport.width;

      // Update container size
      if (containerRef.current) {
        containerRef.current.style.width = `${viewport.width}px`;
        containerRef.current.style.height = `${viewport.height}px`;
      }

      // Render the page
      const renderContext = {
        canvasContext: context,
        viewport: viewport
      };

      await pdfPage.render(renderContext).promise;

      // Render annotations if bbox is provided
      if (bbox && bbox.length === 4) {
        renderAnnotation(bbox, viewport);
      }
    } catch (err: any) {
      console.error('Error rendering page:', err);
      setError(err?.message || 'Failed to render page');
    }
  };

  // Render a single annotation at the specified bbox
  const renderAnnotation = (bboxCoords: number[], viewport: any) => {
    if (!annotationLayerRef.current) return;

    // Clear previous annotations
    annotationLayerRef.current.innerHTML = '';

    // Create highlight element
    const highlight = document.createElement('div');
    highlight.className = 'pdf-annotation';
    highlight.style.position = 'absolute';
    highlight.style.backgroundColor = 'rgba(255, 255, 0, 0.3)'; // Yellow highlight
    highlight.style.border = '2px solid rgba(255, 204, 0, 0.7)';
    highlight.style.borderRadius = '3px';
    
    // Extract coordinates
    const [x1, y1, x2, y2] = bboxCoords;
    
    // Position the highlight
    highlight.style.left = `${x1}px`;
    highlight.style.top = `${y1}px`;
    highlight.style.width = `${x2 - x1}px`;
    highlight.style.height = `${y2 - y1}px`;
    
    // Add to annotation layer
    annotationLayerRef.current.appendChild(highlight);
    
    // Scroll to the annotation
    setTimeout(() => {
      if (containerRef.current) {
        // Calculate center of the annotation
        const centerX = x1 + (x2 - x1) / 2;
        const centerY = y1 + (y2 - y1) / 2;
        
        // Center the annotation in the viewport
        containerRef.current.scrollTo({
          left: centerX - containerRef.current.clientWidth / 2,
          top: centerY - containerRef.current.clientHeight / 2,
          behavior: 'smooth'
        });
      }
    }, 100);
  };

  return (
    <div className="flex flex-col items-center w-full h-full">
      <div 
        ref={containerRef} 
        className="relative mx-auto my-4 shadow-lg overflow-auto"
        style={{ maxHeight: '80vh', maxWidth: '100%' }}
      >
        <canvas ref={canvasRef} className="block" />
        <div 
          ref={annotationLayerRef} 
          className="absolute top-0 left-0 pointer-events-none"
          style={{ width: '100%', height: '100%' }}
        />
      </div>

      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/10">
          <div className="p-4 bg-background rounded-md shadow-lg flex items-center">
            <LoaderIcon className="animate-spin h-5 w-5 mr-2" />
            <span>Loading page {pageNumber}...</span>
          </div>
        </div>
      )}

      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/10">
          <div className="p-4 bg-destructive/10 text-destructive rounded-md shadow-lg">
            <span>{error}</span>
          </div>
        </div>
      )}
    </div>
  );
}