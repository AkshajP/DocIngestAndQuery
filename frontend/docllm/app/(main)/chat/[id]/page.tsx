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
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import DocumentViewerSidebar from '@/components/DocumentViewerSidebar';
import { 
  LoaderIcon, 
  PencilIcon, 
  CheckIcon, 
  XCircleIcon, 
  RefreshCwIcon,
  FileTextIcon,
  XIcon // Added for stop button
} from 'lucide-react';
import { Message } from '@/types/chat';

interface MessageItemProps {
  message: Message;
  onRegenerate?: () => void;
  onViewSource?: (documentId: string, sourceIndex: number, pageNumber: number, bbox: number[]) => void;
  isStreaming?: boolean;
}

export function MessageItem({ message, onRegenerate, onViewSource, isStreaming }: MessageItemProps) {
  const [showSources, setShowSources] = useState(false);
  
  
  return (
    <div className={`py-4 ${message.role === 'assistant' ? '' : 'bg-muted'}`}>
      <div className="max-w-3xl mx-auto px-4">
        <div className="flex items-start gap-2">
          <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
            {message.role === 'user' ? 'U' : 'A'}
          </div>
          <div className="flex-1">
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
              
              {isStreaming && message.role === 'assistant' && (
                <div className="typing-indicator inline-flex items-center mt-1">
                  <span className="dot"></span>
                  <span className="dot"></span>
                  <span className="dot"></span>
                </div>
              )}
            </div>
            
            {message.sources && message.sources.length > 0 && (
  <div className="mt-2">
    <Button 
      variant="ghost" 
      className='rounded-xl '
      size="sm"
      onClick={() => setShowSources(!showSources)}
    >
      {showSources ? (
    <>
      Hide Sources <ChevronUp className="w-4 h-4" />
    </>
  ) : (
    <>
      Show Sources <ChevronDown className="w-4 h-4" />
    </>
  )}
    </Button>
    
    <AnimatePresence>
      {showSources && (
        <motion.div
          key="sources"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.2, ease: 'easeInOut' }}
          className="mt-2 space-y-2 overflow-hidden "
        >
          {message.sources.map((source, index) => {
            const originalBoxes = source.metadata?.original_boxes || [];
            const firstBox = originalBoxes.length > 0 ? originalBoxes[0] : null;
            const pageNumber = firstBox?.original_page_index || 0;
            const bbox = firstBox?.bbox || [0, 0, 0, 0];
            const isOriginalChunk = source.chunk_type === 'original';

            return (
              <div key={index} className="p-2 border rounded-xl text-sm">
                <div className="flex items-center justify-between m-1">
                  <span className="font-medium">
                    {source.chunk_type === 'summary' ? 'Internally Processed Document' : `Source ${index + 1}`}
                  </span>
                  {isOriginalChunk && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() =>
                        onViewSource &&
                        onViewSource(source.document_id, index, pageNumber, bbox)
                      }
                    >
                      <FileTextIcon className="h-4 w-4 mr-1" />
                      View in document
                    </Button>
                  )}
                </div>
                <p className="text-muted-foreground m-1">{source.content}</p>
              </div>
            );
          })}
        </motion.div>
      )}
    </AnimatePresence>
  </div>
)}
              
            
            {message.role === 'assistant' && onRegenerate && !isStreaming && (
              <div className="mt-2">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={onRegenerate}
                >
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
    sendStreamingMessage,
    regenerateResponse,
    updateChatTitle,
    highlightDocumentSource
  } = useChat();
  
  const [chatTitle, setChatTitle] = useState('Loading...');
  const [editingTitle, setEditingTitle] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [inputValue, setInputValue] = useState('');
  const [sending, setSending] = useState(false);
  const [streaming, setStreaming] = useState(false); // New state for streaming
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null); // Track which message is streaming
  const [loadingHighlight, setLoadingHighlight] = useState(false);
  const [highlightUrl, setHighlightUrl] = useState<string | null>(null);
  
  const titleInputRef = useRef<HTMLInputElement>(null);
  const messageEndRef = useRef<HTMLDivElement>(null);
  const streamControlRef = useRef<{close: () => void} | null>(null); // Ref to control stream
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarDocument, setSidebarDocument] = useState<{
    documentId: string;
    pageNumber: number;
    bbox: number[];
    sourceIndex: number;
  } | null>(null);

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

  // Modified to use streaming
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    
    setSending(true);
    try {
      // Use the streaming method instead
      await sendStreamingMessage(chatId, inputValue);
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

  // Add a function to stop streaming
  const handleStopStreaming = () => {
    if (streamControlRef.current) {
      streamControlRef.current.close();
      streamControlRef.current = null;
      setStreaming(false);
      setStreamingMessageId(null);
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

  const handleViewSource = async (
  documentId: string, 
  sourceIndex: number, 
  pageNumber: number,
  bbox: number[]
) => {
  try {
    setLoadingHighlight(true);
    
    // Open the sidebar with the document info
    setSidebarDocument({
      documentId,
      pageNumber: pageNumber + 1, // Convert to 1-based for display
      bbox,
      sourceIndex
    });
    setSidebarOpen(true);
  } catch (error) {
    console.error('Failed to open document viewer:', error);
    toast({
      type: 'error',
      description: 'Failed to open document viewer. Please try again.'
    });
  } finally {
    setLoadingHighlight(false);
  }
};

const handleCloseSidebar = () => {
  setSidebarOpen(false);
  setSidebarDocument(null);
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
      
      <div className={`flex-1 overflow-y-auto transition-all duration-300 ${sidebarOpen ? 'mr-150' : ''}`}>
        <div className="mx-auto max-w-3xl pb-4">
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
                  isStreaming={streamingMessageId === message.id}
                  onRegenerate={message.role === 'assistant' ? () => handleRegenerate(message.id) : undefined}
                  onViewSource={handleViewSource}
                />
              ))}
              <div ref={messageEndRef} />
            </div>
          )}
        </div>
      </div>
      
      <div className={`p-4 bg-background transition-all duration-300 ${sidebarOpen ? 'mr-150' : ''}`}>
        <form onSubmit={handleSendMessage} className="mx-auto max-w-3xl px-4"> {/* Added px-4 to match messages */}
          <div className="flex flex-row gap-2">
            <Textarea
              placeholder="Ask a question about your documents..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              className="resize-none min-h-[40px] max-h-[140px] rounded-2xl"
              disabled={sending}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage(e);
                }
              }}
            />
            <Button 
              type={streaming ? "button" : "submit"} 
              disabled={(!streaming && !inputValue.trim())}
              onClick={streaming ? handleStopStreaming : undefined}
              className='rounded-xl'
            >
              {streaming ? (
                <>
                  <XIcon className="mr-2 h-4 w-4" />
                  Stop
                </>
              ) : sending ? (
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
      <DocumentViewerSidebar
        isOpen={sidebarOpen}
        onClose={handleCloseSidebar}
        documentId={sidebarDocument?.documentId || null}
        pageNumber={sidebarDocument?.pageNumber || 1}
        bbox={sidebarDocument?.bbox}
        sourceIndex={sidebarDocument?.sourceIndex}
      />
    </div>
  );
}