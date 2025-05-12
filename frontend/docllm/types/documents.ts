// types/documents.ts
export enum DocumentStatus {
  PENDING = "pending",
  PROCESSING = "processing",
  FAILED = "failed",
  PROCESSED = "processed",
  DELETED = "deleted"
}

export enum ProcessingStageStatus {
  PENDING = "pending",
  IN_PROGRESS = "in_progress",
  COMPLETED = "completed",
  FAILED = "failed",
  SKIPPED = "skipped"
}

export interface ProcessingStage {
  status: ProcessingStageStatus;
  time_taken?: number;
  started_at?: string;
  completed_at?: string;
  error?: string;
  pages_processed?: number;
  chunks_created?: number;
  embeddings_generated?: number;
  tree_levels?: number;
  progress?: number;
}

export interface DocumentProcessingStages {
  mineru?: ProcessingStage;
  chunking?: ProcessingStage;
  embeddings?: ProcessingStage;
  raptor?: ProcessingStage;
}

export interface DocumentMetadata {
  document_id: string;
  document_name: string;
  case_path: string;
  status: DocumentStatus;
  chunks_count?: number;
  processing_date?: string;
  file_type?: string;
  language?: string;
  content_types?: Record<string, number>;
  raptor_levels?: number[];
}

export interface DocumentProcessRequest {
  document_id?: string;
  original_filename: string;
  metadata?: Record<string, any>;
}

export interface DocumentProcessResponse {
  status: string;
  job_id: string;
  documents: DocumentMetadata[];
  estimated_processing_time: string;
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
}