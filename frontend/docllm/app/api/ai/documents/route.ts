// Update frontend/docllm/app/api/ai/documents/[id]/view/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function GET(
  request: NextRequest,
  context: { params: { id: string } }
) {
  try {
    // Get document ID from context params
    const { id: documentId } = context.params;
    
    // Get the backend base URL
    const backendBaseUrl = process.env.BACKEND_API_URL?.endsWith('/') 
      ? process.env.BACKEND_API_URL.slice(0, -1)
      : process.env.BACKEND_API_URL;
    
    // Direct file access route
    const fileUrl = `${backendBaseUrl}/ai/documents/${documentId}/file`;
    
    console.log(`Fetching document file from: ${fileUrl}`);
    
    // Use a streaming response to properly handle linearized PDFs
    const response = await fetch(fileUrl);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    // Get the content disposition from the backend
    const contentDisposition = response.headers.get('Content-Disposition');
    
    // Create a streaming response
    return new Response(response.body, {
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': contentDisposition || `inline; filename="document-${documentId}.pdf"`,
        // These headers help with streaming
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'public, max-age=3600'
      }
    });
  } catch (error) {
    console.error('Failed to fetch document file:', error);
    return NextResponse.json(
      { error: 'Failed to fetch document file' },
      { status: 500 }
    );
  }
}