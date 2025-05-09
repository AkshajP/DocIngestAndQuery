// frontend/docllm/app/api/ai/documents/[id]/view/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function GET(
  request: NextRequest,
  context: { params: { id: string } }
) {
  try {
    // Get document ID from context params
    const { id: documentId } = await context.params;
    
    // Get the backend base URL, ensuring no double slashes
    const backendBaseUrl = process.env.BACKEND_API_URL?.endsWith('/') 
      ? process.env.BACKEND_API_URL.slice(0, -1)
      : process.env.BACKEND_API_URL;
    
    // Direct file access route - skip metadata lookup for simplicity
    const fileUrl = `${backendBaseUrl}/ai/documents/${documentId}/file`;
    
    console.log(`Fetching document file from: ${fileUrl}`);
    
    const response = await fetch(fileUrl, {
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    // Get the PDF data as an ArrayBuffer
    const pdfData = await response.arrayBuffer();
    
    // Try to get the filename from headers
    const contentDisposition = response.headers.get('Content-Disposition');
    let filename = `document-${documentId}.pdf`;
    
    if (contentDisposition) {
      const matches = /filename="(.+?)"/.exec(contentDisposition);
      if (matches && matches[1]) {
        filename = matches[1];
      }
    }
    
    // Return the PDF content with appropriate headers
    return new NextResponse(pdfData, {
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': `inline; filename="${filename}"`,
      },
    });
  } catch (error) {
    console.error('Failed to fetch document file:', error);
    return NextResponse.json(
      { error: 'Failed to fetch document file' },
      { status: 500 }
    );
  }
}