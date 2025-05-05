// app/(main)/chat/[id]/page.tsx
'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { useChat } from '@/contexts/ChatContext';
import { chatApi } from '@/lib/api';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Messages } from '@/components/messages';
import { toast } from '@/components/toast';
import { LoaderIcon } from '@/components/icons';
import { QueryRequest } from '@/types/chat';

export default function ChatPage() {
  const { id } = useParams();
  const chatId = Array.isArray(id) ? id[0] : id;
  const { messages, loadingMessages, fetchChatMessages, sendMessage } = useChat();
  const [chatTitle, setChatTitle] = useState('Loading...');
  const [inputValue, setInputValue] = useState('');
  const [sending, setSending] = useState(false);

  useEffect(() => {
    const fetchChatDetails = async () => {
      try {
        const chatDetails = await chatApi.getChat(chatId);
        setChatTitle(chatDetails.title);
        fetchChatMessages(chatId);
      } catch (error) {
        toast({
          type: 'error',
          description: 'Failed to load chat details. Please try again.'
        });
      }
    };

    fetchChatDetails();
  }, [chatId, fetchChatMessages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    
    setSending(true);
    try {
      await sendMessage(chatId, inputValue);
      setInputValue('');
    } catch (error) {
      toast({
        type: 'error',
        description: 'Failed to send message. Please try again.'
      });
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="flex flex-col h-screen">
      <header className="border-b p-4">
        <h1 className="text-xl font-bold">{chatTitle}</h1>
      </header>
      
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl">
          {loadingMessages && messages.length === 0 ? (
            <div className="flex justify-center items-center h-full">
              <LoaderIcon className="animate-spin mr-2" />
              <span>Loading messages...</span>
            </div>
          ) : (
            <Messages 
              chatId={chatId}
              messages={messages}
              status={sending ? 'streaming' : 'idle'}
              votes={[]}
              setMessages={() => {}}
              reload={() => fetchChatMessages(chatId)}
              isReadonly={false}
              isArtifactVisible={false}
            />
          )}
        </div>
      </div>
      
      <div className="border-t p-4">
        <form onSubmit={handleSendMessage} className="mx-auto max-w-3xl">
          <div className="flex gap-2">
            <Textarea
              placeholder="Ask a question about your documents..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              className="resize-none min-h-[60px]"
              disabled={sending}
            />
            <Button type="submit" disabled={sending || !inputValue.trim()}>
              {sending ? (
                <>
                  <LoaderIcon className="animate-spin mr-2" />
                  Sending...
                </>
              ) : (
                'Send'
              )}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}