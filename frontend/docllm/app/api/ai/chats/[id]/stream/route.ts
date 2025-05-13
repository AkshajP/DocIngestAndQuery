import { NextRequest } from 'next/server';

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const chatId = params.id;
  const searchParams = request.nextUrl.searchParams;
  const messageId = searchParams.get('messageId') || '';
  
  // Create a ReadableStream to handle SSE
  const stream = new ReadableStream({
    start(controller) {
      // Set up connection to backend streaming endpoint
      const backendUrl = `${process.env.BACKEND_API_URL}/ai/chats/${chatId}/query?stream=true`;
      
      // Forward the request to the backend with EventSource
      const eventSource = new EventSource(backendUrl);
      
      // Forward all SSE events
      eventSource.onmessage = (event) => {
        controller.enqueue(new TextEncoder().encode(`data: ${event.data}\n\n`));
      };
      
      // Forward specific events
      ['start', 'retrieval_complete', 'sources', 'token', 'complete', 'error'].forEach(eventType => {
        eventSource.addEventListener(eventType, (event) => {
          controller.enqueue(new TextEncoder().encode(`event: ${eventType}\ndata: ${event.data}\n\n`));
        });
      });
      
      // Handle close
      eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        eventSource.close();
        controller.close();
      };
      
      // Cleanup function
      return () => {
        eventSource.close();
      };
    }
  });
  
  // Return the streaming response
  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive'
    }
  });
}