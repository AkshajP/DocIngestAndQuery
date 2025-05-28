// Enhanced frontend/docllm/app/(main)/admin/page.tsx with stage-based processing

'use client';

import { useEffect, useState } from 'react';
import { adminApi } from '@/lib/api';
import { Button } from '@/components/ui/button';
import DocumentViewerModal from '@/components/admin/DocumentViewerModal';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { toast } from '@/components/toast';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Eye, RotateCcw, RefreshCw, AlertCircle, CheckCircle, Clock, XCircle, Settings, Trash2 } from 'lucide-react';
import { 
  DocumentMetadata, 
  ProcessingStage, 
  ProcessingStageDetails,
  DocumentProcessingStatus,
  ProcessingStatistics,
  STAGE_DISPLAY_CONFIG,
  STAGE_STATUS_CONFIG,
  ProcessingStageStatus
} from '@/types/documents';

export default function AdminPage() {
  const [stats, setStats] = useState<any>(null);
  const [processingStats, setProcessingStats] = useState<ProcessingStatistics | null>(null);
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [statusCounts, setStatusCounts] = useState<any>({});
  const [activeTab, setActiveTab] = useState('all');
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [retrying, setRetrying] = useState<string | null>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [showStagesModal, setShowStagesModal] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<DocumentMetadata | null>(null);
  const [selectedDocumentStages, setSelectedDocumentStages] = useState<DocumentProcessingStatus | null>(null);
  const [viewerOpen, setViewerOpen] = useState(false);
  const [selectedDocumentForViewer, setSelectedDocumentForViewer] = useState<any>(null);
  const [recovering, setRecovering] = useState<string | null>(null);
  const [systemCleanup, setSystemCleanup] = useState(false);

  const handleOpenViewer = (doc: any) => {
    setSelectedDocumentForViewer(doc);
    setViewerOpen(true);
  };

  useEffect(() => {
    fetchStats();
    fetchDocuments();
    fetchProcessingStats();
  }, []);

  const handleForceUnlock = async (documentId: string) => {
  setRecovering(documentId);
  try {
    const result = await adminApi.forceUnlockDocument(documentId);
    toast({
      type: 'success',
      description: `Document unlocked: ${result.message}`
    });
    fetchDocuments(); // Refresh document list
  } catch (error) {
    console.error('Failed to unlock document:', error);
    toast({
      type: 'error',
      description: 'Failed to unlock document'
    });
  } finally {
    setRecovering(null);
  }
};

const handleSystemCleanup = async () => {
  setSystemCleanup(true);
  try {
    const result = await adminApi.cleanupStaleLocksSystem();
    toast({
      type: 'success',
      description: `System cleanup completed: ${result.message}`
    });
    fetchDocuments();
    fetchStats();
  } catch (error) {
    console.error('Failed to cleanup system:', error);
    toast({
      type: 'error',
      description: 'Failed to cleanup stale locks'
    });
  } finally {
    setSystemCleanup(false);
  }
};


  const fetchStats = async () => {
    try {
      const data = await adminApi.getStats();
      setStats(data);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
      toast({
        type: 'error',
        description: 'Failed to load system statistics'
      });
    }
  };

  const fetchProcessingStats = async () => {
    try {
      const data = await adminApi.getProcessingStatistics();
      setProcessingStats(data);
    } catch (error) {
      console.error('Failed to fetch processing stats:', error);
      // Set null to indicate loading failed
      setProcessingStats(null);
    }
  };

  const fetchDocuments = async (status?: string) => {
    setLoading(true);
    try {
      const data = await adminApi.listDocuments({ status });
      setDocuments(data.documents);
      setStatusCounts(data.status_counts);
      
      // Ensure activeTab is synchronized with the fetched data
      const expectedTab = status || 'all';
      if (activeTab !== expectedTab) {
        setActiveTab(expectedTab);
      }
    } catch (error) {
      console.error('Failed to fetch documents:', error);
      toast({
        type: 'error',
        description: 'Failed to load documents'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (value: string) => {
    // Prevent unnecessary API calls if already on this tab
    if (activeTab === value && !loading) {
      return;
    }
    
    // Update tab state immediately for UI responsiveness
    setActiveTab(value);
    
    // Fetch documents for the new tab
    const statusFilter = value === 'all' ? undefined : value;
    fetchDocuments(statusFilter);
  };

  useEffect(() => {
    const initializeData = async () => {
      await Promise.all([
        fetchStats(),
        fetchDocuments(), // This will load 'all' documents by default
        fetchProcessingStats()
      ]);
    };
    
    initializeData();
  }, []); 

  const handleUpload = async () => {
    if (!uploadFile) return;
    
    setUploading(true);
    try {
      await adminApi.uploadDocument(uploadFile);
      toast({
        type: 'success',
        description: 'Document uploaded successfully. Processing started.'
      });
      setUploadFile(null);
      fetchDocuments();
      fetchStats();
      fetchProcessingStats();
    } catch (error) {
      console.error('Failed to upload document:', error);
      toast({
        type: 'error',
        description: 'Failed to upload document'
      });
    } finally {
      setUploading(false);
    }
  };

  const handleRetry = async (documentId: string, fromStage?: string, forceRestart?: boolean) => {
    setRetrying(documentId);
    try {
      await adminApi.retryDocument(documentId, { fromStage, forceRestart });
      const action = forceRestart ? 'restarted from beginning' : fromStage ? `retried from ${fromStage}` : 'retried';
      toast({
        type: 'success',
        description: `Document processing ${action}`
      });
      fetchDocuments();
      fetchProcessingStats();
    } catch (error) {
      console.error('Failed to retry document processing:', error);
      toast({
        type: 'error',
        description: 'Failed to restart document processing'
      });
    } finally {
      setRetrying(null);
    }
  };

  const handleResetToStage = async (documentId: string, stage: string) => {
    try {
      await adminApi.resetDocumentToStage(documentId, stage);
      toast({
        type: 'success',
        description: `Document reset to ${stage} stage`
      });
      fetchDocuments();
      // Refresh stages modal if open
      if (showStagesModal && selectedDocument?.document_id === documentId) {
        await fetchDocumentStages(documentId);
      }
    } catch (error) {
      console.error('Failed to reset document:', error);
      toast({
        type: 'error',
        description: 'Failed to reset document stage'
      });
    }
  };

  const fetchDocumentStages = async (documentId: string) => {
    if (!documentId) {
      console.error('No document ID provided for stage fetch');
      return;
    }
    
    try {
      const data = await adminApi.getDocumentStages(documentId);
      setSelectedDocumentStages(data || null);
    } catch (error) {
      console.error('Failed to fetch document stages:', error);
      setSelectedDocumentStages(null);
      toast({
        type: 'error',
        description: 'Failed to load document stages'
      });
    }
  };

  const handleOpenStagesModal = async (doc: DocumentMetadata) => {
    if (!doc?.document_id) {
      toast({
        type: 'error',
        description: 'Invalid document selected'
      });
      return;
    }
    
    setSelectedDocument(doc);
    setShowStagesModal(true);
    await fetchDocumentStages(doc.document_id);
  };

  const formatDate = (dateString: string) => {
    if (!dateString) return 'N/A';
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch (e) {
      return dateString;
    }
  };

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case 'processed': return 'bg-green-100 text-green-800 hover:bg-green-200';
      case 'processing': return 'bg-blue-100 text-blue-800 hover:bg-blue-200';
      case 'failed': return 'bg-red-100 text-red-800 hover:bg-red-200';
      case 'pending': return 'bg-yellow-100 text-yellow-800 hover:bg-yellow-200';
      default: return 'bg-gray-100 text-gray-800 hover:bg-gray-200';
    }
  };

  const getStageBadgeClass = (status: ProcessingStageStatus) => {
    const config = STAGE_STATUS_CONFIG[status];
    if (!config) {
      return "bg-gray-100 text-gray-800 hover:opacity-80"; // fallback for unknown status
    }
    return `${config.bgColor} ${config.textColor} hover:opacity-80`;
  };

  const getStageIcon = (stage: ProcessingStage, status: ProcessingStageStatus) => {
    const config = STAGE_DISPLAY_CONFIG[stage];
    if (status === ProcessingStageStatus.COMPLETED) return <CheckCircle className="h-4 w-4 text-green-500" />;
    if (status === ProcessingStageStatus.FAILED) return <XCircle className="h-4 w-4 text-red-500" />;
    if (status === ProcessingStageStatus.IN_PROGRESS) return <Clock className="h-4 w-4 text-blue-500" />;
    return <span className="text-sm">{config?.icon || '❓'}</span>;
  };

  // Enhanced Document Details Modal with Processing State
  const DocumentDetailsModal = ({ document, isOpen, onClose }: { document: DocumentMetadata | null, isOpen: boolean, onClose: () => void }) => {
    if (!isOpen || !document) return null;
    
    const formatContentTypes = (types: Record<string, number>) => {
      if (!types) return "None";
      return Object.entries(types).map(([type, count]) => `${type}: ${count}`).join(", ");
    };
    
    const formatRaptorLevels = (levels: number[]) => {
      if (!levels || !levels.length) return "None";
      return levels.join(", ");
    };
    
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
        <div className="bg-background p-6 rounded-lg shadow-lg max-w-4xl w-full max-h-[80vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold">Document Details</h2>
            <Button variant="ghost" size="sm" onClick={onClose}>×</Button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="space-y-2">
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Document ID</h3>
                <p className="font-mono text-xs break-all">{document.document_id}</p>
              </div>
              
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Filename</h3>
                <p className="break-all">{document.document_name}</p>
              </div>
              
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Status</h3>
                <Badge className={getStatusBadgeClass(document.status)}>
                  {document.status}
                </Badge>
              </div>
            </div>
            
            <div className="space-y-2">
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Pages</h3>
                <p>{document.page_count || 0}</p>
              </div>
              
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Chunks</h3>
                <p>{document.chunks_count || 0}</p>
              </div>
              
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Processing Time</h3>
                <p>{document.total_processing_time ? `${document.total_processing_time.toFixed(2)}s` : 'N/A'}</p>
              </div>
            </div>
          </div>

          {/* Processing State Information */}
          {document.processing_state ? (
            <div className="mb-4">
              <h3 className="text-lg font-semibold mb-2">Processing State</h3>
              <div className="bg-muted p-4 rounded-lg">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-sm text-muted-foreground">Current Stage:</span>
                    <p className="font-medium">
                      {STAGE_DISPLAY_CONFIG[document.processing_state.current_stage]?.label || document.processing_state.current_stage || 'Unknown'}
                    </p>
                  </div>
                  <div>
                    <span className="text-sm text-muted-foreground">Completed Stages:</span>
                    <p className="font-medium">{document.processing_state.completed_stages?.length || 0}</p>
                  </div>
                </div>
                
                {document.processing_state.stage_error_details && Object.keys(document.processing_state.stage_error_details).length > 0 && (
                  <div className="mt-3">
                    <span className="text-sm text-muted-foreground">Recent Errors:</span>
                    <div className="mt-1 space-y-1">
                      {Object.entries(document.processing_state.stage_error_details).map(([stage, error]: [string, any]) => (
                        <div key={stage} className="text-sm text-red-600">
                          <strong>{stage}:</strong> {error?.error || 'Unknown error'}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="mb-4">
              <h3 className="text-lg font-semibold mb-2">Processing State</h3>
              <div className="bg-muted p-4 rounded-lg">
                <p className="text-muted-foreground">No processing state information available</p>
              </div>
            </div>
          )}
          
          <div className="flex justify-between">
            <Button 
              variant="outline" 
              onClick={() => handleOpenStagesModal(document)}
              className="flex items-center gap-2"
            >
              <Settings className="h-4 w-4" />
              View Stages
            </Button>
            <Button variant="outline" onClick={onClose}>Close</Button>
          </div>
        </div>
      </div>
    );
  };

  // New Processing Stages Modal
  const ProcessingStagesModal = ({ 
    document, 
    stages, 
    isOpen, 
    onClose 
  }: { 
    document: DocumentMetadata | null, 
    stages: DocumentProcessingStatus | null, 
    isOpen: boolean, 
    onClose: () => void 
  }) => {
    if (!isOpen || !document || !stages) return null;
    
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
        <div className="bg-background p-6 rounded-lg shadow-lg max-w-4xl w-full max-h-[80vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold">
              Processing Stages - {document.document_name || document.document_id}
            </h2>
            <Button variant="ghost" size="sm" onClick={onClose}>×</Button>
          </div>
          
          <div className="space-y-4">
            {stages?.stages && stages.stages.length > 0 ? (
              stages.stages.map((stage) => {
                const stageConfig = STAGE_DISPLAY_CONFIG[stage.stage];
                const statusConfig = STAGE_STATUS_CONFIG[stage.status];
                
                return (
                  <div key={stage.stage} className={`p-4 rounded-lg border ${stage.is_current ? 'border-blue-500 bg-blue-50' : 'border-gray-200'}`}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        {getStageIcon(stage.stage, stage.status)}
                        <div>
                          <h3 className="font-medium">{stageConfig?.label || stage.stage}</h3>
                          <p className="text-sm text-muted-foreground">{stageConfig?.description || 'Processing stage'}</p>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Badge className={getStageBadgeClass(stage.status)}>
                          {statusConfig?.label || stage.status}
                        </Badge>
                        
                        {stage.retry_count > 0 && (
                          <Badge variant="outline">
                            {stage.retry_count} retries
                          </Badge>
                        )}
                        
                        {stage.is_current && stage.status === ProcessingStageStatus.FAILED && (
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleRetry(document.document_id, stage.stage)}
                            disabled={retrying === document.document_id}
                            className="flex items-center gap-1"
                          >
                            <RefreshCw className="h-3 w-3" />
                            Retry
                          </Button>
                        )}
                        
                        {stage.status === ProcessingStageStatus.COMPLETED && (
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleResetToStage(document.document_id, stage.stage)}
                            className="flex items-center gap-1"
                          >
                            <RotateCcw className="h-3 w-3" />
                            Reset
                          </Button>
                        )}
                      </div>
                    </div>
                    
                    {stage.error_details && (
                      <div className="mt-3 p-3 bg-red-50 rounded border border-red-200">
                        <p className="text-sm text-red-600">
                          <strong>Error:</strong> {stage.error_details.error}
                        </p>
                        <p className="text-xs text-red-500 mt-1">
                          {formatDate(stage.error_details.timestamp)}
                        </p>
                      </div>
                    )}
                    
                    {stage.completion_time && (
                      <div className="mt-2">
                        <p className="text-xs text-muted-foreground">
                          Completed: {formatDate(stage.completion_time)}
                        </p>
                      </div>
                    )}
                  </div>
                );
              })
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                No stage information available
              </div>
            )}
          </div>
          
          <div className="flex justify-between mt-6">
            <div className="flex gap-2">
              <Button 
                variant="outline"
                onClick={() => document && handleRetry(document.document_id, undefined, true)}
                disabled={!document || retrying === document?.document_id}
                className="flex items-center gap-2"
              >
                <RefreshCw className="h-4 w-4" />
                Restart from Beginning
              </Button>
            </div>
            <Button variant="outline" onClick={onClose}>Close</Button>
          </div>
        </div>
      </div>
    );
  };

  useEffect(() => {
  const handleEscapeKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') {
      // Close modals in priority order (top modal first)
      if (viewerOpen) {
        setViewerOpen(false);
      } else if (showStagesModal) {
        setShowStagesModal(false);
      } else if (showDetailsModal) {
        setShowDetailsModal(false);
      }
    }
  };

  document.addEventListener('keydown', handleEscapeKey);
  
  return () => {
    document.removeEventListener('keydown', handleEscapeKey);
  };
}, [viewerOpen, showStagesModal, showDetailsModal]);

  return (
    <div className="w-full min-h-screen px-4 py-8">
      <div className="w-full mx-auto">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold">Admin Dashboard</h1>
        <Button 
          onClick={() => {
            fetchStats();
            fetchDocuments(activeTab === 'all' ? undefined : activeTab);
            fetchProcessingStats();
          }}
        >
          Refresh Data
        </Button>
      </div>
      
      {/* Document Details Modal */}
      <DocumentDetailsModal 
        document={selectedDocument} 
        isOpen={showDetailsModal} 
        onClose={() => setShowDetailsModal(false)}
      />

      {/* Processing Stages Modal */}
      <ProcessingStagesModal
        document={selectedDocument}
        stages={selectedDocumentStages}
        isOpen={showStagesModal}
        onClose={() => setShowStagesModal(false)}
      />
      
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Total Documents</CardTitle>
          </CardHeader>
          <CardContent>
            {stats ? (
              <div className="text-3xl font-bold">
                {stats.document_stats.total_documents}
              </div>
            ) : (
              <div className="h-8 bg-muted animate-pulse rounded"></div>
            )}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Total Pages</CardTitle>
          </CardHeader>
          <CardContent>
            {stats ? (
              <div className="text-3xl font-bold">
                {stats.document_stats.total_pages}
              </div>
            ) : (
              <div className="h-8 bg-muted animate-pulse rounded"></div>
            )}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Total Chunks</CardTitle>
          </CardHeader>
          <CardContent>
            {stats ? (
              <div className="text-3xl font-bold">
                {stats.document_stats.total_chunks}
              </div>
            ) : (
              <div className="h-8 bg-muted animate-pulse rounded"></div>
            )}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Avg. Processing Time</CardTitle>
          </CardHeader>
          <CardContent>
            {stats ? (
              <div className="text-3xl font-bold">
                {stats.document_stats.avg_processing_time.toFixed(2)}s
              </div>
            ) : (
              <div className="h-8 bg-muted animate-pulse rounded"></div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Processing Statistics */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Processing Statistics</CardTitle>
          <CardDescription>Stage-wise processing metrics</CardDescription>
        </CardHeader>
        <CardContent>
          {processingStats ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <h3 className="font-medium mb-2">By Current Stage</h3>
                {processingStats.by_current_stage && Object.keys(processingStats.by_current_stage).length > 0 ? (
                  Object.entries(processingStats.by_current_stage).map(([stage, count]) => (
                    <div key={stage} className="flex justify-between text-sm">
                      <span>{STAGE_DISPLAY_CONFIG[stage as ProcessingStage]?.label || stage}</span>
                      <span>{count}</span>
                    </div>
                  ))
                ) : (
                  <div className="text-sm text-muted-foreground">No data available</div>
                )}
              </div>
              
              <div>
                <h3 className="font-medium mb-2">Stage Errors</h3>
                {processingStats.stage_error_counts && Object.keys(processingStats.stage_error_counts).length > 0 ? (
                  Object.entries(processingStats.stage_error_counts).map(([stage, count]) => (
                    <div key={stage} className="flex justify-between text-sm text-red-600">
                      <span>{STAGE_DISPLAY_CONFIG[stage as ProcessingStage]?.label || stage}</span>
                      <span>{count}</span>
                    </div>
                  ))
                ) : (
                  <div className="text-sm text-muted-foreground">No errors detected</div>
                )}
              </div>
              
              <div>
                <h3 className="font-medium mb-2">Retry Statistics</h3>
                {processingStats.retry_statistics && Object.keys(processingStats.retry_statistics).length > 0 ? (
                  Object.entries(processingStats.retry_statistics).map(([stage, stats]) => (
                    <div key={stage} className="text-sm">
                      <div className="font-medium">{STAGE_DISPLAY_CONFIG[stage as ProcessingStage]?.label || stage}</div>
                      <div className="text-muted-foreground">
                        {stats.total_retries} total retries ({stats.avg_retries_per_doc.toFixed(1)} avg)
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-sm text-muted-foreground">No retries recorded</div>
                )}
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <h3 className="font-medium mb-2">By Current Stage</h3>
                <div className="space-y-2">
                  <div className="h-4 bg-muted animate-pulse rounded"></div>
                  <div className="h-4 bg-muted animate-pulse rounded"></div>
                  <div className="h-4 bg-muted animate-pulse rounded"></div>
                </div>
              </div>
              
              <div>
                <h3 className="font-medium mb-2">Stage Errors</h3>
                <div className="space-y-2">
                  <div className="h-4 bg-muted animate-pulse rounded"></div>
                  <div className="h-4 bg-muted animate-pulse rounded"></div>
                </div>
              </div>
              
              <div>
                <h3 className="font-medium mb-2">Retry Statistics</h3>
                <div className="space-y-2">
                  <div className="h-4 bg-muted animate-pulse rounded"></div>
                  <div className="h-4 bg-muted animate-pulse rounded"></div>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
      
      {/* System Health */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>System Health</CardTitle>
          <CardDescription>Storage and processing metrics</CardDescription>
        </CardHeader>
        <CardContent>
          {stats ? (
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium">Storage Usage</span>
                  <span className="text-sm text-muted-foreground">
                    {stats.system_health.storage_usage.used_gb.toFixed(1)} GB / {stats.system_health.storage_usage.total_gb.toFixed(1)} GB
                  </span>
                </div>
                <Progress value={stats.system_health.storage_usage.usage_percent} className="h-2" />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-sm text-muted-foreground">Vector DB Status</span>
                  <p className="font-medium">{stats.system_health.vector_db_status}</p>
                </div>
                <div>
                  <span className="text-sm text-muted-foreground">Processing Queue</span>
                  <p className="font-medium">{stats.system_health.processing_queue_length} documents waiting</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="h-4 bg-muted animate-pulse rounded mb-4"></div>
              <div className="grid grid-cols-2 gap-4">
                <div className="h-8 bg-muted animate-pulse rounded"></div>
                <div className="h-8 bg-muted animate-pulse rounded"></div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
      
      {/* Document Upload Section */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Upload Document</CardTitle>
          <CardDescription>Upload a new document for processing</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center p-4 border-2 border-dashed rounded-md">
            {uploading ? (
              <div className="flex items-center">
                <div className="animate-spin mr-2 h-5 w-5 border-t-2 border-b-2 border-primary rounded-full"></div>
                <span>Uploading document...</span>
              </div>
            ) : (
              <div className="text-center">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="mx-auto h-12 w-12 text-muted-foreground"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
                {uploadFile ? (
                  <div className="mt-2">
                    <p className="font-medium">{uploadFile.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {(uploadFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                    <div className="flex gap-2 justify-center mt-2">
                      <Button onClick={handleUpload}>
                        Upload and Process
                      </Button>
                      <Button variant="outline" onClick={() => setUploadFile(null)}>
                        Cancel
                      </Button>
                    </div>
                  </div>
                ) : (
                  <>
                    <input
                      type="file"
                      id="file-upload"
                      className="hidden"
                      onChange={(e) => {
                        if (e.target.files && e.target.files[0]) {
                          setUploadFile(e.target.files[0]);
                        }
                      }}
                    />
                    <Button 
                      className="mt-2"
                      onClick={() => {
                        document.getElementById('file-upload')?.click();
                      }}
                    >
                      Select File
                    </Button>
                    <p className="text-sm text-muted-foreground mt-2">
                      Supported formats: PDF
                    </p>
                  </>
                )}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Documents List Section */}
      <Card className="mb-8 w-full">
        <CardHeader>
          <CardTitle>Document Management</CardTitle>
          <CardDescription>View and manage document processing with stage-level control</CardDescription>
        </CardHeader>
        <CardContent className="w-full">
          <Tabs defaultValue="all" value={activeTab} onValueChange={handleTabChange}>
            <TabsList className="mb-4">
              <TabsTrigger value="all">
                All
                {statusCounts && Object.keys(statusCounts).length > 0 && (
                  <span className="ml-1 text-xs">
                    ({Object.values(statusCounts).reduce((sum: number, count: number) => sum + count, 0)})
                  </span>
                )}
              </TabsTrigger>
              <TabsTrigger value="processed">
                Processed
                {statusCounts?.processed !== undefined && (
                  <span className="ml-1 text-xs">({statusCounts.processed})</span>
                )}
              </TabsTrigger>
              <TabsTrigger value="processing">
                Processing
                {statusCounts?.processing !== undefined && (
                  <span className="ml-1 text-xs">({statusCounts.processing})</span>
                )}
              </TabsTrigger>
              <TabsTrigger value="failed">
                Failed
                {statusCounts?.failed !== undefined && (
                  <span className="ml-1 text-xs">({statusCounts.failed})</span>
                )}
              </TabsTrigger>
              <TabsTrigger value="pending">
                Pending
                {statusCounts?.pending !== undefined && (
                  <span className="ml-1 text-xs">({statusCounts.pending})</span>
                )}
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value={activeTab} className="w-full">
              {loading ? (
                <div className="flex justify-center items-center py-8 min-h-[100px] w-full"> {/* Fixed: Added min-height and full width */}
                    <div className="animate-spin mr-2 h-5 w-5 border-t-2 border-b-2 border-primary rounded-full"></div>
                    <span>Loading documents...</span>
                  </div>
              ) : documents.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground min-h-[100px] w-full flex items-center justify-center"> {/* Fixed: Added min-height and centering */}
                    No documents found
                  </div>
              ) : (
                <div className="w-full"> {/* Fixed: Ensure table container is full width */}
                    <div className="overflow-x-auto w-full"> {/* Fixed: Full width overflow container */}
                      <Table className="w-full min-w-full"> 
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-1/6 min-w-[200px] max-w-[150px]">Document ID</TableHead>
                        <TableHead className="w-1/4 min-w-[200px] max-w-[200px]">Filename</TableHead>
                        <TableHead className="w-[100px]">Status</TableHead>
                        <TableHead className="w-[120px]">Current Stage</TableHead>
                        <TableHead className="w-[150px]">Processing Date</TableHead>
                        <TableHead className="w-[100px] text-center">Chunks</TableHead>
                        <TableHead className="w-[80px] text-center">Pages</TableHead>
                        <TableHead className="w-[200px] min-w-[200px]">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {documents.map((doc) => (
                        <TableRow key={doc.document_id}>
                          <TableCell className="font-mono">
                            <Button 
                              variant="link" 
                              className="p-0 h-auto font-mono text-xs text-primary hover:underline"
                              onClick={() => {
                                setSelectedDocument(doc);
                                setShowDetailsModal(true);
                              }}
                            >
                              {doc.document_id}
                            </Button>
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center">
                              <a 
                                href={`/api/ai/documents/${doc.document_id}/view`} 
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center hover:text-primary"
                                title="Click to view document"
                              >
                                <svg
                                  xmlns="http://www.w3.org/2000/svg"
                                  className="mr-2 h-4 w-4"
                                  fill="none"
                                  viewBox="0 0 24 24"
                                  stroke="currentColor"
                                >
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                                  />
                                </svg>
                                <span className="truncate max-w-[200px]" title={doc.document_name}>
                                  {doc.document_name || '(Unnamed)'}
                                </span>
                              </a>
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge className={getStatusBadgeClass(doc.status)}>
                              {doc.status}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            {doc.processing_state?.current_stage ? (
                              <div className="flex items-center gap-1">
                                <span className="text-xs">
                                  {STAGE_DISPLAY_CONFIG[doc.processing_state.current_stage]?.icon || '❓'}
                                </span>
                                <span className="text-sm">
                                  {STAGE_DISPLAY_CONFIG[doc.processing_state.current_stage]?.label || doc.processing_state.current_stage}
                                </span>
                              </div>
                            ) : (
                              <span className="text-muted-foreground text-sm">N/A</span>
                            )}
                          </TableCell>
                          <TableCell>
                            {formatDate(doc.processing_date)}
                          </TableCell>
                          <TableCell className="text-center">
                            {doc.chunks_count || 0}
                          </TableCell>
                          <TableCell className="text-center">
                            {doc.page_count || 0}
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center gap-1">
                              {(doc.status === 'failed' || doc.status === 'processing') && (
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => handleRetry(doc.document_id)}
                                  disabled={retrying === doc.document_id}
                                >
                                  {retrying === doc.document_id ? (
                                    <>
                                      <div className="animate-spin mr-1 h-3 w-3 border-t-2 border-b-2 border-primary rounded-full"></div>
                                      Retrying...
                                    </>
                                  ) : (
                                    <>
                                      <RefreshCw className="h-3 w-3 mr-1" />
                                      Retry
                                    </>
                                  )}
                                </Button>
                              )}
                              
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => handleOpenStagesModal(doc)}
                                className="flex items-center gap-1"
                              >
                                <Settings className="h-3 w-3" />
                                Stages
                              </Button>
                              
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => handleOpenViewer(doc)}
                                className="flex items-center gap-1"
                              >
                                <Eye className="h-3 w-3" />
                                Chunks
                              </Button>
                              {doc.status === 'processing' && (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => handleForceUnlock(doc.document_id)}
                                disabled={recovering === doc.document_id}
                                className="flex items-center gap-1"
                              >
                                {recovering === doc.document_id ? (
                                  <>
                                    <div className="animate-spin mr-1 h-3 w-3 border-t-2 border-b-2 border-primary rounded-full"></div>
                                    Unlocking...
                                  </>
                                ) : (
                                  <>
                                    <AlertCircle className="h-3 w-3" />
                                    Force Unlock
                                  </>
                                )}
                              </Button>
                            )}
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>


      {/*Recovery section */}
      <Card className='outline'>
        <CardHeader>
          <CardTitle>Server Recovery</CardTitle>
          <CardDescription>Recovery tools for server crashes and stuck processes</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <Button
              onClick={handleSystemCleanup}
              disabled={systemCleanup}
              variant="outline"
              className="flex items-center gap-2 bg-red-100 text-red-800 hover:bg-red-200 hover:text-red-800"
            >
              {systemCleanup ? (
                <>
                  <div className="animate-spin mr-2 h-4 w-4 border-t-2 border-b-2 border-primary rounded-full"></div>
                  Cleaning up...
                </>
              ) : (
                <>
                  <RefreshCw className="h-4 w-4" />
                  Cleanup Stale Locks
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
      
      {selectedDocumentForViewer && (
        <DocumentViewerModal
          isOpen={viewerOpen}
          onClose={() => setViewerOpen(false)}
          documentId={selectedDocumentForViewer.document_id}
          documentName={selectedDocumentForViewer.original_filename || 'Unnamed Document'}
        />
      )}
    </div>
    </div>
  );
}