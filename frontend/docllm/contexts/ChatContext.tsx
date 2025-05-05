// contexts/ChatContext.tsx
'use client';

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
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
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export function ChatProvider({ children }: { children: ReactNode }) {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loadingChats, setLoadingChats] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);

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
    try {
      setLoadingMessages(true);
      const response = await chatApi.getChatHistory(chatId);
      setMessages(response.messages);
    } catch (error) {
      console.error('Failed to fetch chat messages:', error);
    } finally {
      setLoadingMessages(false);
    }
  }, []);

  const createChat = useCallback(async (documentIds: string[], title?: string): Promise<string> => {
    try {
      const response = await chatApi.createChat({
        loaded_documents: documentIds.map(id => ({ document_id: id, title: '' })),
        title: title || 'New Chat',
      });
      await fetchChats();
      return response.id;
    } catch (error) {
      console.error('Failed to create chat:', error);
      throw error;
    }
  }, [fetchChats]);

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
        top_k: 5,
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