// contexts/ChatContext.tsx - Updated with correct regenerate API calls
'use client';

import React, { createContext, useContext, useState, useCallback, ReactNode, useRef} from 'react';
import { chatApi } from '@/lib/api';
import { ChatSummary, Message, QueryRequest, ChatSettings, DEFAULT_CHAT_SETTINGS, MessageRole } from '@/types/chat';
import { toast } from '@/components/toast';

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
  sendStreamingMessage: (chatId: string, question: string) => Promise<void>;
  updateChatTitle: (chatId: string, title: string) => Promise<void>;
  deleteChat: (chatId: string) => Promise<void>;
  regenerateResponse: (chatId: string, messageId: string, settings?: any) => Promise<void>;
  streamRegenerateResponse: (chatId: string, messageId: string, settings?: any) => Promise<void>;
  getChatTitle: (chatId: string) => Promise<string>;
  highlightDocumentSource: (chatId: string, messageId: string, documentId: string) => Promise<string>;
  cancelStreaming: () => void;
  chatSettings: ChatSettings;
  updatingSettings: boolean;
  loadChatSettings: (chatId: string) => Promise<void>;
  updateChatSettings: (chatId: string, settings: ChatSettings) => Promise<void>;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export function ChatProvider({ children }: { children: ReactNode }) {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loadingChats, setLoadingChats] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamController, setStreamController] = useState<{ close: () => void } | null>(null);
  const [chatSettings, setChatSettings] = useState<ChatSettings>(DEFAULT_CHAT_SETTINGS);
  const [updatingSettings, setUpdatingSettings] = useState(false);

  const streamControlRef = useRef(null);

  const loadChatSettings = async (chatId: string) => {
    try {
      const response = await chatApi.getChatSettings(chatId);
      setChatSettings(response.settings);
    } catch (error) {
      console.error('Failed to load chat settings:', error);
      setChatSettings(DEFAULT_CHAT_SETTINGS);
    }
  };

  const updateChatSettings = async (chatId: string, settings: ChatSettings) => {
    setUpdatingSettings(true);
    try {
      await chatApi.updateChatSettings(chatId, settings);
      setChatSettings(settings);
      toast({
        type: 'success',
        description: 'Settings updated successfully!'
      });
    } catch (error) {
      console.error('Failed to update chat settings:', error);
      toast({
        type: 'error',
        description: 'Failed to update settings. Please try again.'
      });
      throw error;
    } finally {
      setUpdatingSettings(false);
    }
  };

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
      console.log("line57,chatContext")
      
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
        role: MessageRole.USER,
        content: question,
        created_at: new Date().toISOString(),
      };
      setMessages(prev => [...prev, tempMessage]);
  
      const queryRequest: QueryRequest = {
        question,
        use_tree: false,
        top_k: 10,
      };
  
      // Make sure you're using the correct function here
      const response = await chatApi.submitQuery(chatId, queryRequest);
      console.log("chatContext chatAPi res line 139->", response)

  
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
  
  // FIXED: Correct regenerate response implementation (non-streaming)
  const regenerateResponse = useCallback(async (
    chatId: string, 
    messageId: string, 
    settings?: any
  ) => {
    try {
      setLoadingMessages(true);
      
      // Use the correct regenerate API endpoint
      const response = await chatApi.regenerateResponse(chatId, messageId, settings);
      
      // Refresh messages to get the updated response
      await fetchChatMessages(chatId);
    } catch (error) {
      console.error('Failed to regenerate response:', error);
      throw error;
    } finally {
      setLoadingMessages(false);
    }
  }, [fetchChatMessages]);

  // NEW: Streaming regenerate response
  const streamRegenerateResponse = useCallback(async (
    chatId: string, 
    messageId: string, 
    settings?: any
  ) => {
    if (!chatId || !messageId) return;
    
    try {
      // Find the message being regenerated and clear its content
      setMessages(prev => 
        prev.map(msg => 
          msg.id === messageId 
            ? { 
                ...msg, 
                content: '', 
                sources: undefined,
                processing_stats: { streaming: true }
              } 
            : msg
        )
      );
      
      setStreaming(true);
      
      const controller = await chatApi.streamRegenerateResponse(
        chatId,
        messageId,
        settings || {},
        {
          onStart: (data) => {
            console.log('Regeneration started:', data);
          },
          
          onToken: (token) => {
            // Update the message content with new token
            setMessages(prev => 
              prev.map(msg => 
                msg.id === messageId 
                  ? { ...msg, content: msg.content + token } 
                  : msg
              )
            );
          },
          
          onSources: (sources) => {
            // Add sources to the message
            setMessages(prev => 
              prev.map(msg => 
                msg.id === messageId 
                  ? { ...msg, sources } 
                  : msg
              )
            );
          },
          
          onComplete: (data) => {
            // Update message with final stats
            setMessages(prev => 
              prev.map(msg => 
                msg.id === messageId 
                  ? { 
                      ...msg, 
                      processing_stats: {
                        streaming: false,
                        time_taken: data.time_taken,
                        token_count: data.token_count
                      }
                    } 
                  : msg
              )
            );
            
            setStreaming(false);
            setStreamController(null);
            
            // Fetch the official messages from the server after completion
            fetchChatMessages(chatId);
          },
          
          onError: (error) => {
            console.error('Regeneration stream error:', error);
            
            // Update message to show error state
            setMessages(prev => 
              prev.map(msg => 
                msg.id === messageId 
                  ? { 
                      ...msg, 
                      content: 'Error regenerating response. Please try again.',
                      processing_stats: { streaming: false, error: true }
                    } 
                  : msg
              )
            );
            
            setStreaming(false);
            setStreamController(null);
            
            // Show error toast
            toast({
              type: 'error',
              description: error.error || 'Error regenerating response'
            });
          }
        }
      );
      
      setStreamController(controller);
    } catch (error) {
      console.error('Failed to start regeneration streaming:', error);
      setStreaming(false);
      
      // Show error toast
      toast({
        type: 'error',
        description: 'Failed to regenerate response. Please try again.'
      });
    }
  }, [fetchChatMessages]);
  
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

  const sendStreamingMessage = async (chatId: string, question: string) => {
    if (!chatId || !question.trim()) return;
    
    // Create temporary IDs for messages
    const tempUserMsgId = `user-${Date.now()}`;
    const tempAssistantMsgId = `assistant-${Date.now()}`;
    
    // Add user message to state
    const userMessage: Message = {
      id: tempUserMsgId,
      role: MessageRole.USER,
      content: question,
      created_at: new Date().toISOString(),
    };
    
    // Create placeholder for assistant message
    const assistantMessage: Message = {
      id: tempAssistantMsgId,
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString(),
      processing_stats: {
        streaming: true
      }
    };
    
    // Update messages with user message and empty assistant message
    setMessages(prev => [...prev, userMessage, assistantMessage]);
    setStreaming(true);
    
    try {
      const controller = await chatApi.streamQuery(
        chatId,
        question,
        {
          onToken: (token) => {
            // Update assistant message with new token
            setMessages(prev => 
              prev.map(msg => 
                msg.id === tempAssistantMsgId 
                  ? { ...msg, content: msg.content + token } 
                  : msg
              )
            );
          },
          
          onSources: (sources) => {
            // Add sources to the assistant message
            setMessages(prev => 
              prev.map(msg => 
                msg.id === tempAssistantMsgId 
                  ? { ...msg, sources } 
                  : msg
              )
            );
          },
          
          onComplete: (data) => {
            // Update assistant message with final stats
            setMessages(prev => 
              prev.map(msg => 
                msg.id === tempAssistantMsgId 
                  ? { 
                      ...msg, 
                      processing_stats: {
                        streaming: false,
                        time_taken: data.time_taken,
                        token_count: data.token_count
                      }
                    } 
                  : msg
              )
            );
            
            setStreaming(false);
            setStreamController(null);
            
            // Fetch the official messages from the server after completion
            fetchChatMessages(chatId);
          },
          
          onError: (error) => {
            console.error('Stream error:', error);
            
            // Remove the temporary messages
            setMessages(prev => 
              prev.filter(msg => 
                msg.id !== tempUserMsgId && msg.id !== tempAssistantMsgId
              )
            );
            
            setStreaming(false);
            setStreamController(null);
            
            // Show error toast
            toast({
              type: 'error',
              description: error.error || 'Error streaming response'
            });
          }
        }
      );
      
      setStreamController(controller);
    } catch (error) {
      console.error('Failed to start streaming:', error);
      
      // Remove the temporary messages
      setMessages(prev => 
        prev.filter(msg => 
          msg.id !== tempUserMsgId && msg.id !== tempAssistantMsgId
        )
      );
      
      setStreaming(false);
      
      // Show error toast
      toast({
        type: 'error',
        description: 'Failed to send message. Please try again.'
      });
    }
  };
  
  
  // Add a function to cancel streaming
  const cancelStreaming = () => {
    if (streamController) {
      streamController.close();
      setStreamController(null);
      setStreaming(false);
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
      sendStreamingMessage,
      updateChatTitle,
      deleteChat,
      regenerateResponse,
      streamRegenerateResponse, // NEW: Export streaming regenerate
      getChatTitle,
      highlightDocumentSource,
      cancelStreaming,
      chatSettings,
      updatingSettings,
      loadChatSettings,
      updateChatSettings,
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