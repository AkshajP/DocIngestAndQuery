// frontend/docllm/app/(main)/chat/[id]/page.tsx
'use client';

import { useEffect, useState, useRef } from 'react';
import { useParams } from 'next/navigation';
import { useChat } from '@/contexts/ChatContext';
import { chatApi } from '@/lib/api';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { toast } from '@/components/toast';
import { 
  LoaderIcon, 
  PencilIcon, 
  CheckIcon, 
  XCircleIcon, 
  RefreshCwIcon,
  FileTextIcon
} from 'lucide-react';
import { Message } from '@/types/chat';

interface MessageItemProps {
  message: Message;
  onRegenerate?: () => void;
  onViewSource?: (documentId: string) => void;
}

function MessageItem({ message, onRegenerate, onViewSource }: MessageItemProps) {
  const [showSources, setShowSources] = useState(false);
  
  return (
    <div className={`py-4 ${message.role === 'assistant' ? 'bg-muted/30' : ''}`}>
      <div className="max-w-3xl mx-auto px-4">
        <div className="flex items-start gap-2">
          <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
            {message.role === 'user' ? 'U' : 'A'}
          </div>
          <div className="flex-1">
            <div className="prose">
              {message.content}
            </div>
            
            {message.sources && message.sources.length > 0 && (
              <div className="mt-2">
                <Button 
                  variant="ghost" 
                  size="sm"
                  onClick={() => setShowSources(!showSources)}
                >
                  {showSources ? 'Hide Sources' : 'Show Sources'}
                </Button>
                
                {showSources && (
                  <div className="mt-2 space-y-2">
                    {message.sources.map((source, index) => (
                      <div key={index} className="p-2 border rounded-md text-sm">
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-medium">Source {index + 1}</span>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => onViewSource && onViewSource(source.document_id)}
                          >
                            <FileTextIcon className="h-4 w-4 mr-1" />
                            View in document
                          </Button>
                        </div>
                        <p className="text-muted-foreground">{source.content}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            
            {message.role === 'assistant' && onRegenerate && (
              <div className="mt-2">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={onRegenerate}
                >
                  <RefreshCwIcon className="h-4 w-4 mr-1" />
                  Regenerate response
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ChatPage() {
  const { id } = useParams();
  const chatId = Array.isArray(id) ? id[0] : id;
  const { 
    messages, 
    loadingMessages, 
    fetchChatMessages, 
    sendMessage,
    regenerateResponse,
    updateChatTitle,
    highlightDocumentSource
  } = useChat();
  
  const [chatTitle, setChatTitle] = useState('Loading...');
  const [editingTitle, setEditingTitle] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [inputValue, setInputValue] = useState('');
  const [sending, setSending] = useState(false);
  const [loadingHighlight, setLoadingHighlight] = useState(false);
  const [highlightUrl, setHighlightUrl] = useState<string | null>(null);
  
  const titleInputRef = useRef<HTMLInputElement>(null);
  const messageEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchChatDetails = async () => {
      try {
        const chatDetails = await chatApi.getChat(chatId);
        setChatTitle(chatDetails.title);
        fetchChatMessages(chatId);
      } catch (error) {
        console.error('Failed to load chat details:', error);
        toast({
          type: 'error',
          description: 'Failed to load chat details. Please try again.'
        });
      }
    };

    fetchChatDetails();
  }, [chatId, fetchChatMessages]);

  useEffect(() => {
    // Scroll to bottom when new messages arrive
    if (messageEndRef.current) {
      messageEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleTitleEdit = () => {
    setNewTitle(chatTitle);
    setEditingTitle(true);
    setTimeout(() => titleInputRef.current?.focus(), 0);
  };

  const handleTitleSave = async () => {
    if (newTitle.trim() && newTitle !== chatTitle) {
      try {
        await updateChatTitle(chatId, newTitle.trim());
        setChatTitle(newTitle.trim());
      } catch (error) {
        console.error('Failed to update chat title:', error);
        toast({
          type: 'error',
          description: 'Failed to update chat title. Please try again.'
        });
      }
    }
    setEditingTitle(false);
  };

  const handleTitleCancel = () => {
    setEditingTitle(false);
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    
    setSending(true);
    try {
      await sendMessage(chatId, inputValue);
      setInputValue('');
    } catch (error) {
      console.error('Failed to send message:', error);
      toast({
        type: 'error',
        description: 'Failed to send message. Please try again.'
      });
    } finally {
      setSending(false);
    }
  };

  const handleRegenerate = async (messageId: string) => {
    try {
      await regenerateResponse(chatId, messageId);
    } catch (error) {
      console.error('Failed to regenerate response:', error);
      toast({
        type: 'error',
        description: 'Failed to regenerate response. Please try again.'
      });
    }
  };

  const handleViewSource = async (documentId: string, messageId: string) => {
    try {
      setLoadingHighlight(true);
      const url = await highlightDocumentSource(chatId, messageId, documentId);
      setHighlightUrl(url);
      // Here you would show a modal or viewer for the highlighted PDF
      window.open(url, '_blank');
    } catch (error) {
      console.error('Failed to load document highlight:', error);
      toast({
        type: 'error',
        description: 'Failed to load document highlight. Please try again.'
      });
    } finally {
      setLoadingHighlight(false);
    }
  };

  return (
    <div className="flex flex-col h-screen">
      <header className="border-b p-4 flex items-center justify-between">
        {editingTitle ? (
          <div className="flex items-center gap-2 flex-1">
            <Input 
              ref={titleInputRef}
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              className="max-w-md"
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleTitleSave();
                if (e.key === 'Escape') handleTitleCancel();
              }}
            />
            <Button size="sm" variant="ghost" onClick={handleTitleSave}>
              <CheckIcon className="h-4 w-4" />
            </Button>
            <Button size="sm" variant="ghost" onClick={handleTitleCancel}>
              <XCircleIcon className="h-4 w-4" />
            </Button>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold">{chatTitle}</h1>
            <Button size="sm" variant="ghost" onClick={handleTitleEdit}>
              <PencilIcon className="h-4 w-4" />
            </Button>
          </div>
        )}
      </header>
      
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl pb-20">
          {loadingMessages && messages.length === 0 ? (
            <div className="flex justify-center items-center h-full">
              <LoaderIcon className="animate-spin mr-2" />
              <span>Loading messages...</span>
            </div>
          ) : (
            <div>
              {messages.map(message => (
                <MessageItem 
                  key={message.id}
                  message={message}
                  onRegenerate={message.role === 'assistant' ? () => handleRegenerate(message.id) : undefined}
                  onViewSource={(documentId) => handleViewSource(documentId, message.id)}
                />
              ))}
              {sending && (
                <div className="py-4 bg-muted/30">
                  <div className="max-w-3xl mx-auto px-4">
                    <div className="flex items-center">
                      <LoaderIcon className="animate-spin mr-2" />
                      <span>Generating response...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messageEndRef} />
            </div>
          )}
        </div>
      </div>
      
      <div className="border-t p-4 fixed bottom-0 left-0 right-0 bg-background">
        <form onSubmit={handleSendMessage} className="mx-auto max-w-3xl">
          <div className="flex gap-2">
            <Textarea
              placeholder="Ask a question about your documents..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              className="resize-none min-h-[60px]"
              disabled={sending}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage(e);
                }
              }}
            />
            <Button type="submit" disabled={sending || !inputValue.trim()}>
              {sending ? (
                <>
                  <LoaderIcon className="animate-spin mr-2 h-4 w-4" />
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