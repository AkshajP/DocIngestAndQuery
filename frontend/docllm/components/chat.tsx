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
    
    try {
      // Use chatApi.streamQuery instead of manual fetch
      chatApi.streamQuery(
        id,
        input.trim(),
        {}, // options
        {
          onStart: (data) => {
            console.log('Stream started', data);
          },
          onSources: (sources) => {
            // Store sources for later attachment to the complete message
            setMessages(prev => prev.map(msg => 
              msg.id === assistantMessageId 
                ? { ...msg, sources } 
                : msg
            ));
          },
          onToken: (token) => {
            // Update the message in real-time
            setMessages(prev => prev.map(msg => 
              msg.id === assistantMessageId 
                ? { ...msg, content: (msg.content || '') + token } 
                : msg
            ));
          },
          onComplete: (data) => {
            // Update with final message and metadata
            setMessages(prev => prev.map(msg => 
              msg.id === assistantMessageId 
                ? { 
                    ...msg, 
                    processing_stats: {
                      time_taken: data.time_taken,
                      token_count: data.token_count,
                      // other stats
                    }
                  } 
                : msg
            ));
            
            setStreamingMessageId(undefined);
          },
          onError: (error) => {
            console.error('Stream error:', error);
            toast({
              type: 'error',
              description: error.error || 'Error streaming response'
            });
            
            setStreamingMessageId(undefined);
          }
        }
      );
    } catch (error) {
      console.error('Failed to send streaming message:', error);
      toast({
        type: 'error',
        description: 'Failed to send message. Please try again.'
      });
      setStreamingMessageId(undefined);
    }
  };

  const handleCancelStreaming = () => {
    if (status === 'streaming') {
      stop();
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