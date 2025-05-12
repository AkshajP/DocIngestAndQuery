// contexts/ChatContext.tsx
'use client';

import React, { createContext, useContext, useState, useCallback, ReactNode, useRef} from 'react';
import { chatApi } from '@/lib/api';
import { ChatSummary, Message, QueryRequest } from '@/types/chat';

interface ChatContextType {
  chats: ChatSummary[];
  loadingChats: boolean;
  loadingMessages: boolean;
  messages: Message[];
  fetchChats: () => Promise<void>;
  fetchChatMessages: (chatId: string) => Promise<void>;
  createChat: (documentIds: string[], title?: string) => Promise<string>;
  updateChatDocuments: (chatId: string, add: string[], remove: string[]) => Promise<void>;
  sendMessage: (chatId: string, question: string) => Promise<void>;
  updateChatTitle: (chatId: string, title: string) => Promise<void>;
  deleteChat: (chatId: string) => Promise<void>;
  regenerateResponse: (chatId: string, messageId: string) => Promise<void>;
  getChatTitle: (chatId: string) => Promise<string>;
  highlightDocumentSource: (chatId: string, messageId: string, documentId: string) => Promise<string>;

}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export function ChatProvider({ children }: { children: ReactNode }) {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loadingChats, setLoadingChats] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStreamedMessage, setCurrentStreamedMessage] = useState('');
  const streamControlRef = useRef(null);

  const fetchChats = useCallback(async () => {
    try {
      setLoadingChats(true);
      const response = await chatApi.listChats();
      setChats(response.chats);
    } catch (error) {
      console.error('Failed to fetch chats:', error);
    } finally {
      setLoadingChats(false);
    }
  }, []);

  const fetchChatMessages = useCallback(async (chatId: string) => {
    if (!chatId) {
      console.warn('Cannot fetch messages: No chat ID provided');
      return;
    }
  
    try {
      setLoadingMessages(true);
      
      // Try to get the chat details first to verify it exists
      try {
        const chatDetails = await chatApi.getChat(chatId);
        // If we get here, the chat exists and we can proceed
      } catch (chatError) {
        console.warn(`Chat ${chatId} not found or not accessible, skipping message fetch`);
        setLoadingMessages(false);
        return; // Exit early if chat doesn't exist
      }
      
      // Now fetch the messages
      const response = await chatApi.getChatHistory(chatId);
      setMessages(response.messages || []);
    } catch (error) {
      console.error('Failed to fetch chat messages:', error);
      setMessages([]); // Set empty messages on error
    } finally {
      setLoadingMessages(false);
    }
  }, []);

  const createChat = useCallback(async (documentIds: string[], title?: string): Promise<string> => {
    try {
      // First create the chat
      const response = await chatApi.createChat({
        loaded_documents: documentIds.map(id => ({ document_id: id, title: '' })),
        title: title || 'New Chat',
      });
      
      // Add to local chat list without immediately trying to fetch messages
      setChats(prev => [
        {
          id: response.id,
          title: response.title,
          messages_count: 0,
          last_active: new Date().toISOString()
        },
        ...prev
      ]);
      
      // Return the chat ID so navigation can happen
      return response.id;
      
      // Don't call fetchChats() or fetchChatMessages() here
      // Let the chat page handle loading messages when it mounts
    } catch (error) {
      console.error('Failed to create chat:', error);
      throw error;
    }
  }, []);

  const updateChatDocuments = useCallback(async (chatId: string, add: string[], remove: string[]) => {
    try {
      await chatApi.updateChatDocuments(chatId, { add, remove });
    } catch (error) {
      console.error('Failed to update chat documents:', error);
      throw error;
    }
  }, []);

  const sendMessage = useCallback(async (chatId: string, question: string) => {
    try {
      setLoadingMessages(true);
      // Create temporary message
      const tempMessage: Message = {
        id: `temp-${Date.now()}`,
        role: 'user',
        content: question,
        created_at: new Date().toISOString(),
      };
      setMessages(prev => [...prev, tempMessage]);

      const queryRequest: QueryRequest = {
        question,
        use_tree: false,
        top_k: 10,
      };

      const response = await chatApi.submitQuery(chatId, queryRequest);

      // Update messages with real message and response
      fetchChatMessages(chatId);
    } catch (error) {
      console.error('Failed to send message:', error);
      // Remove temp message on error
      setMessages(prev => prev.filter(m => !m.id.startsWith('temp-')));
    } finally {
      setLoadingMessages(false);
    }
  }, [fetchChatMessages]);

  const updateChatTitle = useCallback(async (chatId: string, title: string) => {
    try {
      await chatApi.updateChat(chatId, { title });
      
      // Create a new array to ensure React detects the state change
      const updatedChats = chats.map(chat => 
        chat.id === chatId ? { ...chat, title } : chat
      );
      
      // Set the entire chats array to trigger re-renders in all components using it
      setChats(updatedChats);
      
      // Also fetch fresh data from the server to ensure everything is in sync
      fetchChats();
    } catch (error) {
      console.error('Failed to update chat title:', error);
      throw error;
    }
  }, [chats, fetchChats]);
  
  const deleteChat = useCallback(async (chatId: string) => {
    try {
      await chatApi.deleteChat(chatId);
      // Update local state
      setChats(prev => prev.filter(chat => chat.id !== chatId));
    } catch (error) {
      console.error('Failed to delete chat:', error);
      throw error;
    }
  }, []);
  
  const regenerateResponse = useCallback(async (chatId: string, messageId: string) => {
    try {
      setLoadingMessages(true);
      // Find the previous message to get the question
      const questionIndex = messages.findIndex(m => m.id === messageId) - 1;
      if (questionIndex >= 0) {
        const question = messages[questionIndex].content;
        const response = await chatApi.submitQuery(chatId, { question });
        // Refresh messages
        await fetchChatMessages(chatId);
      }
    } catch (error) {
      console.error('Failed to regenerate response:', error);
    } finally {
      setLoadingMessages(false);
    }
  }, [messages, fetchChatMessages]);
  
  const getChatTitle = useCallback(async (chatId: string) => {
    try {
      const chat = await chatApi.getChat(chatId);
      return chat.title;
    } catch (error) {
      console.error('Failed to get chat title:', error);
      return "Untitled Chat";
    }
  }, []);
  
  const highlightDocumentSource = useCallback(async (chatId: string, messageId: string, documentId: string) => {
    try {
      // This would call the API endpoint for highlighting
      // For now we'll return a placeholder since the API is not fully implemented in the frontend
      return `/api/ai/chats/${chatId}/messages/${messageId}/highlights/${documentId}`;
    } catch (error) {
      console.error('Failed to get document highlights:', error);
      throw error;
    }
  }, []);

  const sendStreamingMessage = async (chatId, question) => {
    // First add the user message to the UI immediately
    const userMessageId = `user-${Date.now()}`;
    const assistantMessageId = `assistant-${Date.now()}`;
    
    // Add user message to messages
    const userMessage = {
      id: userMessageId,
      role: 'user',
      content: question,
      created_at: new Date().toISOString()
    };
    
    // Create a placeholder for assistant message
    const assistantMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString(),
      isStreaming: true
    };
    
    // Update messages state with both messages
    setMessages(prev => [...prev, userMessage, assistantMessage]);
    // setStreamingMessageId(assistantMessageId);
    // Start streaming
    setIsStreaming(true);
    setCurrentStreamedMessage('');
    
    try {
      // Call our streaming API function
      streamControlRef.current = chatApi.streamQuery(
        chatId,
        question,
        {}, // options
        {
          onStart: (data) => {
            console.log('Stream started', data);
          },
          onSources: (sources) => {
            // Store sources to later attach to the complete message
            setMessages(prev => prev.map(msg => 
              msg.id === assistantMessageId 
                ? { ...msg, sources } 
                : msg
            ));
          },
          onToken: (token) => {
            // Append token to current streamed message
            setCurrentStreamedMessage(prev => prev + token);
            
            // Update the message in real-time
            setMessages(prev => prev.map(msg => 
              msg.id === assistantMessageId 
                ? { ...msg, content: prev + token } 
                : msg
            ));
          },
          onComplete: (data) => {
            // Update with final message and metadata
            setMessages(prev => prev.map(msg => 
              msg.id === assistantMessageId 
                ? { 
                    ...msg, 
                    isStreaming: false,
                    processing_stats: {
                      time_taken: data.time_taken,
                      tokens_used: data.token_count,
                      // other stats
                    }
                  } 
                : msg
            ));
            
            setIsStreaming(false);
            streamControlRef.current = null;
          },
          onError: (error) => {
            console.error('Stream error:', error);
            toast({
              type: 'error',
              description: error.error || 'Error streaming response'
            });
            
            setIsStreaming(false);
            streamControlRef.current = null;
          }
        }
      );
    } catch (error) {
      console.error('Failed to send streaming message:', error);
      toast({
        type: 'error',
        description: 'Failed to send message. Please try again.'
      });
      setIsStreaming(false);
    }
  };
  
  // Add a function to cancel streaming
  const cancelStreaming = () => {
    if (streamControlRef.current) {
      streamControlRef.current.close();
      streamControlRef.current = null;
      setIsStreaming(false);
    }
  };
  

  return (
    <ChatContext.Provider value={{
      chats,
      loadingChats,
      loadingMessages,
      messages,
      fetchChats,
      fetchChatMessages,
      createChat,
      updateChatDocuments,
      sendMessage,
      updateChatTitle,
      deleteChat,
      regenerateResponse,
      getChatTitle,
      highlightDocumentSource,
      sendStreamingMessage,
      cancelStreaming
    }}>
      {children}
    </ChatContext.Provider>
  );
}


export function useChat() {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
}
