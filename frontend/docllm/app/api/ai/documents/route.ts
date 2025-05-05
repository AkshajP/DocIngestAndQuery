// app/api/ai/documents/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const limit = searchParams.get('limit') || '10';
  const offset = searchParams.get('offset') || '0';
  
  try {
    // Forward to your backend
    const response = await fetch(`${process.env.BACKEND_API_URL}/ai/chats/documents?limit=${limit}&offset=${offset}`, {
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