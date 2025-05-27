export enum DocumentStatus {
  PENDING = "pending",
  PROCESSING = "processing", 
  FAILED = "failed",
  PROCESSED = "processed",
  DELETED = "deleted"
}

export enum ProcessingStage {
  UPLOAD = "upload",
  EXTRACTION = "extraction", 
  CHUNKING = "chunking",
  EMBEDDING = "embedding",
  TREE_BUILDING = "tree_building",
  VECTOR_STORAGE = "vector_storage",
  COMPLETED = "completed"
}

export enum ProcessingStageStatus {
  PENDING = "pending",
  IN_PROGRESS = "in_progress", 
  COMPLETED = "completed",
  FAILED = "failed",
  SKIPPED = "skipped"
}

export interface ProcessingStageDetails {
  stage: ProcessingStage;
  status: ProcessingStageStatus;
  is_current: boolean;
  completion_time?: string;
  error_details?: {
    error: string;
    timestamp: string;
  };
  retry_count: number;
}

export interface ProcessingState {
  current_stage: ProcessingStage;
  completed_stages: ProcessingStage[];
  stage_completion_times: Record<string, string>;
  stage_error_details: Record<string, any>;
  retry_counts: Record<string, number>;
  last_updated: string;
}

export interface DocumentProcessingStatus {
  document_id: string;
  current_stage: ProcessingStage;
  last_updated: string;
  stages: ProcessingStageDetails[];
}

export interface DocumentMetadata {
  document_id: string;
  document_name: string;  
  status: DocumentStatus;
  chunks_count?: number;
  processing_date?: string;
  file_type?: string;
  language?: string;
  page_count?: number;
  content_types?: Record<string, number>;
  raptor_levels?: number[];
  case_path?: string;
  
  // Enhanced processing information
  processing_state?: ProcessingState;
  total_processing_time?: number;
  failure_stage?: string;
  error_message?: string;
  retry_count?: number;
}

export interface ProcessingStatistics {
  total_documents: number;
  by_status: Record<string, number>;
  by_current_stage: Record<string, number>;
  stage_completion_counts: Record<string, number>;
  stage_error_counts: Record<string, number>;
  retry_statistics: Record<string, {
    total_retries: number;
    avg_retries_per_doc: number;
    max_retries: number;
    documents_with_retries: number;
  }>;
  last_updated: string;
}

export interface DocumentProcessRequest {
  document_id?: string;
  original_filename: string;
  metadata?: Record<string, any>;
  force_restart?: boolean;
}

export interface DocumentProcessResponse {
  status: string;
  document_id: string;
  case_id: string;
  processing_time: number;
  stored_file_path: string;
  doc_dir: string;
  processing_state: DocumentProcessingStatus;
  chunks_count?: number;
  tree_nodes_count?: number;
  raptor_levels?: number[];
  error?: string;
  failed_stage?: string;
}

export interface DocumentListResponse {
  documents: DocumentMetadata[];
  pagination: {
    total: number;
    limit: number;
    offset: number;
  };
}

export interface DocumentDetailResponse {
  document_id: string;
  document_name: string;
  status: DocumentStatus;
  chunks_count: number;
  processing_stats: Record<string, any>;
  content_types?: Record<string, number>;
  raptor_levels?: number[];
  language?: string;
  processing_date?: string;
  processing_state?: ProcessingState;
}

// Stage display configurations
export const STAGE_DISPLAY_CONFIG = {
  [ProcessingStage.UPLOAD]: {
    label: "File Upload",
    description: "Document uploaded and stored",
    icon: "📄",
    color: "blue"
  },
  [ProcessingStage.EXTRACTION]: {
    label: "Content Extraction",
    description: "OCR and content extraction from PDF",
    icon: "🔍",
    color: "purple"
  },
  [ProcessingStage.CHUNKING]: {
    label: "Content Chunking", 
    description: "Breaking content into chunks",
    icon: "✂️",
    color: "orange"
  },
  [ProcessingStage.EMBEDDING]: {
    label: "Embedding Generation",
    description: "Generating vector embeddings",
    icon: "🧠",
    color: "green"
  },
  [ProcessingStage.TREE_BUILDING]: {
    label: "RAPTOR Tree Building",
    description: "Building hierarchical tree structure",
    icon: "🌳",
    color: "brown"
  },
  [ProcessingStage.VECTOR_STORAGE]: {
    label: "Vector Storage",
    description: "Storing in vector database",
    icon: "💾",
    color: "indigo"
  },
  [ProcessingStage.COMPLETED]: {
    label: "Completed",
    description: "All processing completed successfully",
    icon: "✅",
    color: "green"
  }
};

export const STAGE_STATUS_CONFIG = {
  [ProcessingStageStatus.PENDING]: {
    label: "Pending",
    color: "gray",
    bgColor: "bg-gray-100",
    textColor: "text-gray-800"
  },
  [ProcessingStageStatus.IN_PROGRESS]: {
    label: "In Progress",
    color: "blue", 
    bgColor: "bg-blue-100",
    textColor: "text-blue-800"
  },
  [ProcessingStageStatus.COMPLETED]: {
    label: "Completed",
    color: "green",
    bgColor: "bg-green-100", 
    textColor: "text-green-800"
  },
  [ProcessingStageStatus.FAILED]: {
    label: "Failed",
    color: "red",
    bgColor: "bg-red-100",
    textColor: "text-red-800"
  },
  [ProcessingStageStatus.SKIPPED]: {
    label: "Skipped", 
    color: "yellow",
    bgColor: "bg-yellow-100",
    textColor: "text-yellow-800"
  }
};