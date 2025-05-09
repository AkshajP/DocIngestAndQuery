// frontend/docllm/app/api/ai/admin/stats/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    const response = await fetch(`${process.env.BACKEND_API_URL}/ai/admin/stats`, {
      headers: {
        'Content-Type': 'application/json',
        // Add auth headers as needed
      },
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Failed to fetch admin stats:', error);
    return NextResponse.json(
      { error: 'Failed to fetch admin stats' },
      { status: 500 }
    );
  }
}

// frontend/docllm/app/api/ai/admin/documents/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const status = searchParams.get('status');
  
  const url = status 
    ? `${process.env.BACKEND_API_URL}/ai/admin/documents?status=${status}`
    : `${process.env.BACKEND_API_URL}/ai/admin/documents`;
  
  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        // Add auth headers as needed
      },
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Failed to fetch documents:', error);
    return NextResponse.json(
      { error: 'Failed to fetch documents' },
      { status: 500 }
    );
  }
}

// frontend/docllm/app/api/ai/admin/documents/upload/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    
    const response = await fetch(`${process.env.BACKEND_API_URL}/ai/admin/documents/upload`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Failed to upload document:', error);
    return NextResponse.json(
      { error: 'Failed to upload document' },
      { status: 500 }
    );
  }
}

// frontend/docllm/app/api/ai/admin/documents/[id]/retry/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function POST(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const documentId = params.id;
    
    const response = await fetch(
      `${process.env.BACKEND_API_URL}/ai/admin/documents/${documentId}/retry`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Add auth headers as needed
        },
      }
    );
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Failed to retry document processing:', error);
    return NextResponse.json(
      { error: 'Failed to retry document processing' },
      { status: 500 }
    );
  }
}