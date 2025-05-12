'use client';

import type { Attachment, UIMessage } from 'ai';
import { useChat } from '@ai-sdk/react';
import { useEffect, useState } from 'react';
import useSWR, { useSWRConfig } from 'swr';
import { ChatHeader } from '@/components/chat-header';
import type { Vote } from '@/lib/db/schema';
import { fetcher, generateUUID } from '@/lib/utils';
import { Artifact } from './artifact';
import { MultimodalInput } from './multimodal-input';
import { Messages } from './messages';
import type { VisibilityType } from './visibility-selector';
import { useArtifactSelector } from '@/hooks/use-artifact';
import { unstable_serialize } from 'swr/infinite';
import { getChatHistoryPaginationKey } from './sidebar-history';
import { toast } from './toast';
import type { Session } from 'next-auth';
import { useSearchParams } from 'next/navigation';
import { useChatVisibility } from '@/hooks/use-chat-visibility';

export function Chat({
  id,
  initialMessages,
  initialChatModel,
  initialVisibilityType,
  isReadonly,
  session,
  autoResume,
}: {
  id: string;
  initialMessages: Array<UIMessage>;
  initialChatModel: string;
  initialVisibilityType: VisibilityType;
  isReadonly: boolean;
  session: Session;
  autoResume: boolean;
}) {
  const { mutate } = useSWRConfig();
  const [streamingMessageId, setStreamingMessageId] = useState<string | undefined>(undefined);
  
  const { visibilityType } = useChatVisibility({
    chatId: id,
    initialVisibilityType,
  });

  const {
    messages,
    setMessages,
    handleSubmit,
    input,
    setInput,
    append,
    status,
    stop,
    reload,
    experimental_resume,
  } = useChat({
    id,
    initialMessages,
    experimental_throttle: 100,
    sendExtraMessageFields: true,
    generateId: generateUUID,
    experimental_prepareRequestBody: (body) => ({
      id,
      message: body.messages.at(-1),
      selectedChatModel: initialChatModel,
      selectedVisibilityType: visibilityType,
    }),
    onFinish: () => {
      setStreamingMessageId(undefined); // Clear streaming ID when finished
      mutate(unstable_serialize(getChatHistoryPaginationKey));
    },
    onError: (error) => {
      setStreamingMessageId(undefined); // Clear streaming ID on error
      toast({
        type: 'error',
        description: error.message,
      });
    },
  });

  // Add a custom submit handler that manages streaming state
  const handleStreamingSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    
    if (!input.trim()) return;
    
    // Add the user message immediately
    const userMessageId = generateUUID();
    append({
      id: userMessageId,
      role: 'user',
      content: input.trim(),
    });
    
    // Prepare for streaming response
    const assistantMessageId = generateUUID();
    setStreamingMessageId(assistantMessageId);
    
    // Add placeholder assistant message
    append({
      id: assistantMessageId,
      role: 'assistant',
      content: '', // Empty content that will be filled by streaming
    });
    
    setInput('');
    
    // Connect directly to streaming endpoint
    let eventSource: EventSource | null = null;
    let accumulatedContent = '';
    
    try {
      // Generate a URL with the streaming parameter
      const streamUrl = new URL(`/api/ai/chats/${id}/query`, window.location.origin);
      streamUrl.searchParams.append('stream', 'true');
      
      // Create headers for EventSource - needed for POST request
      const headers = new Headers();
      headers.append('Content-Type', 'application/json');
      
      // Create a controller to be able to abort the connection
      const controller = new AbortController();
      
      // Setup the request payload
      const payload = JSON.stringify({
        question: input.trim(),
        use_tree: false, // Set your defaults or get from state
        top_k: 10, // Set your defaults or get from state
      });
      
      // Since EventSource only supports GET by default, we need to use fetch with ReadableStream
      const response = await fetch(streamUrl, {
        method: 'POST',
        headers: headers,
        body: payload,
        signal: controller.signal
      });
      
      if (!response.ok || !response.body) {
        throw new Error(`Server error: ${response.status}`);
      }
      
      // Setup stream processing
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      // Process the stream
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        // Decode chunk
        const chunk = decoder.decode(value, { stream: true });
        
        // Process SSE format (split by double newlines)
        const events = chunk.split('\n\n');
        for (const event of events) {
          if (!event.trim()) continue;
          
          try {
            // Extract event type and data
            const eventTypeMatch = event.match(/^event: (.+)$/m);
            const dataMatch = event.match(/^data: (.+)$/m);
            
            if (eventTypeMatch && dataMatch) {
              const eventType = eventTypeMatch[1];
              const dataStr = dataMatch[1];
              
              if (eventType === 'token') {
                // Parse token data
                const data = JSON.parse(dataStr);
                accumulatedContent += data.token;
                
                // Update message with new content
                setMessages(messages => 
                  messages.map(msg => 
                    msg.id === assistantMessageId 
                      ? { ...msg, content: accumulatedContent } 
                      : msg
                  )
                );
              } 
              else if (eventType === 'error') {
                // Handle error events
                const data = JSON.parse(dataStr);
                toast({
                  type: 'error',
                  description: data.error || 'An error occurred',
                });
                controller.abort();
                break;
              }
              else if (eventType === 'complete') {
                // Handle completion
                console.log('Stream completed successfully');
                break;
              }
            }
          } catch (err) {
            console.error('Error processing stream event:', event, err);
          }
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
      setStreamingMessageId(undefined);
      toast({
        type: 'error',
        description: error instanceof Error ? error.message : 'Failed to stream response',
      });
    } finally {
      // Clean up and reset state
      setStreamingMessageId(undefined);
    }
  };
  
  // Pass the streaming state to Messages component
  return (
    <>
      <div className="flex flex-col min-w-0 h-dvh bg-background">
        <ChatHeader
          chatId={id}
          selectedModelId={initialChatModel}
          selectedVisibilityType={initialVisibilityType}
          isReadonly={isReadonly}
          session={session}
        />

        <Messages
          chatId={id}
          status={status}
          votes={votes}
          messages={messages}
          setMessages={setMessages}
          reload={reload}
          isReadonly={isReadonly}
          isArtifactVisible={isArtifactVisible}
          streamingMessageId={streamingMessageId} // Pass streaming message ID
        />

        <form 
          className="flex mx-auto px-4 bg-background pb-4 md:pb-6 gap-2 w-full md:max-w-3xl"
          onSubmit={handleStreamingSubmit} // Use our custom submit handler
        >
          {!isReadonly && (
            <MultimodalInput
              chatId={id}
              input={input}
              setInput={setInput}
              handleSubmit={handleStreamingSubmit} // Use our custom submit handler
              status={status}
              stop={() => {
                stop();
                setStreamingMessageId(undefined);
              }}
              attachments={attachments}
              setAttachments={setAttachments}
              messages={messages}
              setMessages={setMessages}
              append={append}
              selectedVisibilityType={visibilityType}
              isStreaming={!!streamingMessageId} // Pass streaming state
            />
          )}
        </form>
      </div>

      <Artifact
        chatId={id}
        input={input}
        setInput={setInput}
        handleSubmit={handleStreamingSubmit} // Use our custom submit handler
        status={status}
        stop={() => {
          stop();
          setStreamingMessageId(undefined);
        }}
        attachments={attachments}
        setAttachments={setAttachments}
        append={append}
        messages={messages}
        setMessages={setMessages}
        reload={reload}
        votes={votes}
        isReadonly={isReadonly}
        selectedVisibilityType={visibilityType}
      />
    </>
  );
}