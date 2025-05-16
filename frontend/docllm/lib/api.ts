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
    },
  
    getDocumentDetails: async (documentId: string): Promise<any> => {
      const response = await fetch(`/api/ai/documents/${documentId}`);
      if (!response.ok) {
        throw new Error("Failed to fetch document details");
      }
      return response.json();
    },
  
    getPageUrl: (documentId: string, pageNumber: number): string => {
      return `/api/ai/documents/${documentId}/page/${pageNumber}`;
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
      // Explicitly set stream=false to ensure we get a JSON response
      return fetchWithErrorHandling<QueryResponse>(`/chats/${chatId}/query?stream=false`, {
        method: "POST",
        body: JSON.stringify(data),
      });
    },
    
    streamQuery: async (
      chatId: string, 
      question: string,
      callbacks: {
        onStart?: (data: any) => void,
        onToken?: (token: string) => void,
        onSources?: (sources: any[]) => void,
        onComplete?: (data: any) => void,
        onError?: (error: any) => void
      }
    ) => {
      const controller = new AbortController();
      const signal = controller.signal;
      
      try {
        // Make POST request with streaming response
        const response = await fetch(`${API_BASE_URL}/chats/${chatId}/query?stream=true`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            question: question,
            use_tree: false,
            top_k: 10
          }),
          signal: signal
        });
        
        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }
        
        if (!response.body) {
          throw new Error("Response body stream not available");
        }
        
        // Set up stream processing
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        // Process the stream
        const processStream = async () => {
          try {
            while (true) {
              const { value, done } = await reader.read();
              
              if (done) {
                // Process any remaining data in buffer
                if (buffer.trim()) {
                  processEventData(buffer);
                }
                return;
              }
              
              // Decode and add to buffer
              const chunk = decoder.decode(value, { stream: true });
              buffer += chunk;
              
              // Process any complete events in the buffer
              const events = buffer.split('\n\n');
              buffer = events.pop() || ''; // Keep the last potentially incomplete event in buffer
              
              for (const event of events) {
                processEventData(event);
              }
            }
          } catch (err) {
            if (err.name !== 'AbortError') {
              callbacks.onError?.({ error: err.message });
            }
          }
        };
        
        // Helper to process event data
        const processEventData = (eventText: string) => {
          if (!eventText.trim()) return;
          
          // Extract event type and data
          const eventTypeMatch = eventText.match(/^event: (.+)$/m);
          const dataMatch = eventText.match(/^data: (.+)$/m);
          
          if (!eventTypeMatch || !dataMatch) return;
          
          const eventType = eventTypeMatch[1];
          const dataText = dataMatch[1];
          
          try {
            const data = JSON.parse(dataText);
            
            switch (eventType) {
              case 'start':
                callbacks.onStart?.(data);
                break;
                
              case 'token':
                callbacks.onToken?.(data.token);
                break;
                
              case 'sources':
                callbacks.onSources?.(data.sources);
                break;
                
              case 'complete':
                callbacks.onComplete?.(data);
                break;
                
              case 'error':
                callbacks.onError?.(data);
                break;
            }
          } catch (err) {
            console.error('Error parsing event data:', err, eventText);
          }
        };
        
        // Start processing the stream
        processStream();
        
        // Return control object
        return {
          close: () => {
            controller.abort();
          }
        };
      } catch (error) {
        callbacks.onError?.({ error: error.message });
        return {
          close: () => {}
        };
      }
    }};
  
  export const adminApi = {
    getStats: async (): Promise<any> => {
      return fetchWithErrorHandling<any>(`/admin/stats`);
    },
    
    listDocuments: async (params?: { status?: string }): Promise<any> => {
      const queryParams = new URLSearchParams();
      if (params?.status) queryParams.append("status", params.status);
      
      const queryString = queryParams.toString() ? `?${queryParams.toString()}` : "";
      return fetchWithErrorHandling<any>(`/admin/documents${queryString}`);
    },
    
    uploadDocument: async (file: File, caseId?: string): Promise<any> => {
      const formData = new FormData();
      formData.append("file", file);
      if (caseId) formData.append("case_id", caseId);
      
      const response = await fetch(`${API_BASE_URL}/admin/documents/upload`, {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `API error: ${response.status}`);
      }
      
      return await response.json();
    },
    
    retryDocument: async (documentId: string): Promise<any> => {
      return fetchWithErrorHandling<any>(`/admin/documents/${documentId}/retry`, {
        method: "POST",
      });
    },
    getDocumentChunks: async (documentId: string, options?: { 
      chunkType?: string, 
      pageNumber?: number 
    }): Promise<any> => {
      const queryParams = new URLSearchParams();
      
      if (options?.chunkType) {
        queryParams.append("chunk_type", options.chunkType);
      }
      
      if (options?.pageNumber !== undefined) {
        queryParams.append("page_number", options.pageNumber.toString());
      }
      
      const queryString = queryParams.toString() ? `?${queryParams.toString()}` : "";
      
      return fetchWithErrorHandling<any>(`/admin/documents/${documentId}/chunks${queryString}`);
    }
  };