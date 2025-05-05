// types/chat.ts
export enum MessageRole {
    USER = "user",
    ASSISTANT = "assistant",
    SYSTEM = "system"
  }
  
  export interface ChatDocument {
    document_id: string;
    title: string;
  }
  
  export interface Message {
    id: string;
    role: MessageRole;
    content: string;
    created_at: string;
    sources?: any[];
    processing_stats?: Record<string, any>;
    token_count?: number;
    input_tokens?: number;
    output_tokens?: number;
    model_used?: string;
    response_time?: number;
  }
  
  export interface Chat {
    id: string;
    title: string;
    created_at: string;
    updated_at?: string;
    messages_count: number;
    loaded_documents: ChatDocument[];
  }
  
  export interface ChatSummary {
    id: string;
    title: string;
    messages_count: number;
    last_active?: string;
  }
  
  export interface ChatCreateRequest {
    loaded_documents?: ChatDocument[];
    title?: string;
    settings?: Record<string, any>;
  }
  
  export interface ChatListResponse {
    chats: ChatSummary[];
    pagination: {
      total: number;
      limit: number;
      offset: number;
    };
  }
  
  export interface ChatDetailResponse {
    id: string;
    title: string;
    messages_count: number;
    loaded_documents: ChatDocument[];
    history: {
      messages: Message[];
    };
  }
  
  export interface ChatUpdateRequest {
    title?: string;
  }
  
  export interface ChatDocumentsUpdateRequest {
    add?: string[];
    remove?: string[];
  }
  
  export interface ChatHistoryResponse {
    messages: Message[];
    pagination: {
      total: number;
      limit: number;
      offset: number;
    };
  }
  
  export interface QueryRequest {
    question: string;
    use_tree?: boolean;
    top_k?: number;
    tree_level_filter?: number[];
    model_override?: string;
  }
  
  export interface QuerySource {
    document_id: string;
    content: string;
    score: number;
    metadata: Record<string, any>;
    original_boxes?: any[];
    tree_level?: number;
  }
  
  export interface QueryResponse {
    id: string;
    answer: string;
    sources: QuerySource[];
    processing_stats: {
      time_taken: number;
      input_tokens: number;
      output_tokens: number;
      total_tokens: number;
      retrieval_time?: number;
      llm_time?: number;
      method: string;
      model_used?: string;
    };
  }