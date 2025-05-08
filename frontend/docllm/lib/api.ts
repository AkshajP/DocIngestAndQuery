// lib/api.ts
import {
    DocumentListResponse,
    DocumentDetailResponse,
    ChatListResponse,
    ChatDetailResponse,
    ChatCreateRequest,
    ChatDocumentsUpdateRequest,
    QueryRequest,
    QueryResponse,
    ChatHistoryResponse
  } from "@/types";
  
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "/api";
  
  async function fetchWithErrorHandling<T>(
    url: string,
    options?: RequestInit
  ): Promise<T> {
    try {
      const response = await fetch(`${API_BASE_URL}${url}`, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...options?.headers,
        },
      });
  
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `API error: ${response.status}`);
      }
  
      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${url}`, error);
      throw error;
    }
  }
  
  // Document APIs
  export const documentApi = {
    listDocuments: async (params?: { limit?: number; offset?: number }): Promise<DocumentListResponse> => {
      const queryParams = new URLSearchParams();
      if (params?.limit) queryParams.append("limit", params.limit.toString());
      if (params?.offset) queryParams.append("offset", params.offset.toString());
      
      const queryString = queryParams.toString() ? `?${queryParams.toString()}` : "";
      return fetchWithErrorHandling<DocumentListResponse>(`/documents/${queryString}`);
    },
    
    getDocument: async (documentId: string): Promise<DocumentDetailResponse> => {
      return fetchWithErrorHandling<DocumentDetailResponse>(`/documents/${documentId}`);
    }
  };
  
  // Chat APIs
  export const chatApi = {
    createChat: async (data: ChatCreateRequest): Promise<ChatDetailResponse> => {
      return fetchWithErrorHandling<ChatDetailResponse>(`/chats`, {
        method: "POST",
        body: JSON.stringify(data),
      });
    }
    ,
    
    listChats: async (params?: { limit?: number; offset?: number }): Promise<ChatListResponse> => {
      const queryParams = new URLSearchParams();
      if (params?.limit) queryParams.append("limit", params.limit.toString());
      if (params?.offset) queryParams.append("offset", params.offset.toString());
      
      const queryString = queryParams.toString() ? `?${queryParams.toString()}` : "";
      return fetchWithErrorHandling<ChatListResponse>(`/chats${queryString}`);
    },
    
    getChat: async (chatId: string): Promise<ChatDetailResponse> => {
      return fetchWithErrorHandling<ChatDetailResponse>(`/chats/${chatId}`);
    },
    
    updateChat: async (chatId: string, data: { title: string }): Promise<ChatDetailResponse> => {
      return fetchWithErrorHandling<ChatDetailResponse>(`/chats/${chatId}`, {
        method: "PATCH",
        body: JSON.stringify(data),
      });
    },
    
    deleteChat: async (chatId: string): Promise<{ status: string; message: string }> => {
      return fetchWithErrorHandling<{ status: string; message: string }>(`/chats/${chatId}`, {
        method: "DELETE",
      });
    },
    
    getChatHistory: async (chatId: string, params?: { limit?: number; offset?: number }): Promise<ChatHistoryResponse> => {
      const queryParams = new URLSearchParams();
      if (params?.limit) queryParams.append("limit", params.limit.toString());
      if (params?.offset) queryParams.append("offset", params.offset.toString());
      
      const queryString = queryParams.toString() ? `?${queryParams.toString()}` : "";
      return fetchWithErrorHandling<ChatHistoryResponse>(`/chats/${chatId}/history${queryString}`);
    },
    
    updateChatDocuments: async (chatId: string, data: ChatDocumentsUpdateRequest): Promise<{ status: string; loaded_documents: any[] }> => {
      return fetchWithErrorHandling<{ status: string; loaded_documents: any[] }>(`/chats/${chatId}/documents`, {
        method: "PUT",
        body: JSON.stringify(data),
      });
    },
    
    submitQuery: async (chatId: string, data: QueryRequest): Promise<QueryResponse> => {
      return fetchWithErrorHandling<QueryResponse>(`/chats/${chatId}/query`, {
        method: "POST",
        body: JSON.stringify(data),
      });
    },
    
    streamQuery: async (chatId: string, data: QueryRequest): Promise<Response> => {
      // This function returns the raw Response object for streaming
      const response = await fetch(`${API_BASE_URL}/chats/${chatId}/query?stream=true`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `API error: ${response.status}`);
      }
      
      return response;
    }
  };