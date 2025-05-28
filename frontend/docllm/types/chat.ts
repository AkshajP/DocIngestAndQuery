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

  export interface ChatSettings {
  // Search method settings
  use_tree_search: boolean;
  use_hybrid_search: boolean;
  vector_weight: number; // 0.0 to 1.0
  
  // Retrieval settings
  top_k: number; // 1 to 50
  tree_level_filter: number[] | null; // null means all levels
  content_types: string[] | null;
  
  // Model settings
  llm_model: string | null;
  
  // UI preferences
  show_sources: boolean;
  auto_scroll: boolean;
}
export const DEFAULT_CHAT_SETTINGS: ChatSettings = {
  use_tree_search: false,
  use_hybrid_search: true,
  vector_weight: 0.5,
  top_k: 10,
  tree_level_filter: null,
  content_types: null,
  llm_model: null,
  show_sources: true,
  auto_scroll: true,
};

// Preset configurations for common use cases
export const SETTINGS_PRESETS = {
  balanced: {
    ...DEFAULT_CHAT_SETTINGS,
    use_hybrid_search: true,
    vector_weight: 0.5,
    top_k: 10,
  },
  precise: {
    ...DEFAULT_CHAT_SETTINGS,
    use_hybrid_search: true,
    vector_weight: 0.7, // More weight on vector similarity
    top_k: 5,
    tree_level_filter: [0], // Only original chunks
  },
  comprehensive: {
    ...DEFAULT_CHAT_SETTINGS,
    use_tree_search: true,
    use_hybrid_search: true,
    vector_weight: 0.4,
    top_k: 15,
  },
  keyword_focused: {
    ...DEFAULT_CHAT_SETTINGS,
    use_hybrid_search: true,
    vector_weight: 0.2, // More weight on BM25/keyword matching
    top_k: 12,
  }
};
  
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