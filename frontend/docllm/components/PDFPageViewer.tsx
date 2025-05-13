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
  const renderTaskRef = useRef<any>(null);

  // Load the PDF document
  useEffect(() => {
    let isActive = true;
    
    const loadPdf = async () => {
      try {
        setLoading(true);
        
        // Create a URL for the PDF
        const pdfUrl = `/api/ai/documents/${documentId}/view`;
        
        // Configure the loading options to enable streaming
        const loadingTask = pdfjs.getDocument({
          url: pdfUrl,
          rangeChunkSize: 65536, // 64KB chunks for streaming
          disableAutoFetch: false, // Allow auto fetching
          disableStream: false // Enable streaming
        });
        
        const document = await loadingTask.promise;
        
        // Check if component is still mounted
        if (!isActive) return;
        
        setPdf(document);
        onLoaded?.(document.numPages);
        setLoading(false);
      } catch (err: any) {
        console.error('Error loading PDF:', err);
        if (isActive) {
          setError(err?.message || 'Failed to load PDF');
          setLoading(false);
        }
      }
    };

    loadPdf();

    // Cleanup function
    return () => {
      isActive = false;
    };
  }, [documentId, onLoaded]);

  // Handle page changes
  useEffect(() => {
    let isActive = true;
    
    const loadPage = async () => {
      if (!pdf) return;
      
      try {
        setLoading(true);
        
        // Cancel any ongoing rendering
        if (renderTaskRef.current) {
          renderTaskRef.current.cancel();
          renderTaskRef.current = null;
        }
        
        // Cleanup previous page if exists
        if (page) {
          page.cleanup();
        }
        
        // Check for valid page number
        if (pageNumber < 1 || pageNumber > pdf.numPages) {
          throw new Error(`Invalid page number: ${pageNumber}. Document has ${pdf.numPages} pages.`);
        }
        
        // Load the requested page
        const pdfPage = await pdf.getPage(pageNumber);
        
        // Check if component is still mounted
        if (!isActive) return;
        
        setPage(pdfPage);
        
        // Render the page
        await renderPage(pdfPage, zoom);
        
        setLoading(false);
      } catch (err: any) {
        console.error('Error loading page:', err);
        if (isActive) {
          setError(err?.message || 'Failed to load page');
          setLoading(false);
        }
      }
    };
    
    if (pdf) {
      loadPage();
    }
    
    // Cleanup function
    return () => {
      isActive = false;
      // Cancel any ongoing rendering
      if (renderTaskRef.current) {
        renderTaskRef.current.cancel();
        renderTaskRef.current = null;
      }
    };
  }, [pdf, pageNumber]);

  // Re-render when zoom changes
  useEffect(() => {
    if (page) {
      renderPage(page, zoom);
    }
  }, [zoom]);

  const renderPage = async (pdfPage: PDFPageProxy, scale: number) => {
    if (!canvasRef.current) return;
  
    try {
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
  
      if (!context) {
        throw new Error('Canvas context is null');
      }
  
      // Cancel any ongoing rendering
      if (renderTaskRef.current) {
        renderTaskRef.current.cancel();
        renderTaskRef.current = null;
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
  
      // Render the page and store the rendering task
      const renderContext = {
        canvasContext: context,
        viewport: viewport
      };
  
      renderTaskRef.current = pdfPage.render(renderContext);
      
      try {
        // Wait for rendering to complete
        await renderTaskRef.current.promise;
      } catch (err) {
        // Specifically check for rendering cancelled exception
        if (err instanceof Error && 
            (err.message === 'Rendering cancelled' || 
             err.name === 'RenderingCancelledException')) {
          console.log('Render cancelled, this is normal during page navigation');
          return; // Exit early without error
        }
        // Rethrow other errors
        throw err;
      }
      
      renderTaskRef.current = null;
  
      // Render annotations if bbox is provided
      if (bbox && bbox.length === 4) {
        renderAnnotation(bbox, viewport);
      }
    } catch (err: any) {
      // Skip rendering cancelled errors
      if (err instanceof Error && 
          (err.message === 'Rendering cancelled' || 
           err.name === 'RenderingCancelledException')) {
        return;
      }
      
      console.error('Error rendering page:', err);
      setError(err?.message || 'Failed to render page');
    }
  };

  // Render annotations
  const renderAnnotation = (bboxCoords: number[], viewport: any) => {
    if (!annotationLayerRef.current) return;

    // Clear previous annotations
    annotationLayerRef.current.innerHTML = '';

    // Create highlight element
    const highlight = document.createElement('div');
    highlight.className = 'pdf-annotation';
    highlight.style.position = 'absolute';
    highlight.style.backgroundColor = 'rgba(255, 255, 0, 0.3)';
    highlight.style.border = '2px solid rgba(255, 204, 0, 0.7)';
    highlight.style.borderRadius = '3px';
    
    // Extract coordinates
    const [x1, y1, x2, y2] = bboxCoords;
    
    // Scale coordinates based on viewport scale
    const scaledX1 = x1 * viewport.scale;
    const scaledY1 = y1 * viewport.scale;
    const scaledX2 = x2 * viewport.scale;
    const scaledY2 = y2 * viewport.scale;
    
    // Position the highlight
    highlight.style.left = `${scaledX1}px`;
    highlight.style.top = `${scaledY1}px`;
    highlight.style.width = `${scaledX2 - scaledX1}px`;
    highlight.style.height = `${scaledY2 - scaledY1}px`;
    
    // Add to annotation layer
    annotationLayerRef.current.appendChild(highlight);
    
    // Add pulsing effect to make the annotation more noticeable
    highlight.style.animation = 'pulse-highlight 2s infinite';
    
    // Add the animation style if it doesn't exist
    if (!document.getElementById('highlight-animation')) {
      const style = document.createElement('style');
      style.id = 'highlight-animation';
      style.textContent = `
        @keyframes pulse-highlight {
          0% { box-shadow: 0 0 0 0 rgba(255, 204, 0, 0.7); }
          70% { box-shadow: 0 0 0 10px rgba(255, 204, 0, 0); }
          100% { box-shadow: 0 0 0 0 rgba(255, 204, 0, 0); }
        }
      `;
      document.head.appendChild(style);
    }
    
    // Scroll to the annotation
    setTimeout(() => {
      if (containerRef.current) {
        // Calculate center of the annotation
        const centerX = scaledX1 + (scaledX2 - scaledX1) / 2;
        const centerY = scaledY1 + (scaledY2 - scaledY1) / 2;
        
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