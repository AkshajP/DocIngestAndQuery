// frontend/docllm/app/(main)/admin/page.tsx
'use client';

import { useEffect, useState } from 'react';
import { adminApi } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { toast } from '@/components/toast';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';

export default function AdminPage() {
  const [stats, setStats] = useState<any>(null);
  const [documents, setDocuments] = useState<any[]>([]);
  const [statusCounts, setStatusCounts] = useState<any>({});
  const [activeTab, setActiveTab] = useState('all');
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [retrying, setRetrying] = useState<string | null>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<any>(null);

  useEffect(() => {
    fetchStats();
    fetchDocuments();
  }, []);

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

  const fetchDocuments = async (status?: string) => {
    setLoading(true);
    try {
      const data = await adminApi.listDocuments({ status });
      setDocuments(data.documents);
      setStatusCounts(data.status_counts);
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
    setActiveTab(value);
    fetchDocuments(value === 'all' ? undefined : value);
  };

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
      // Refresh documents list
      fetchDocuments();
      fetchStats();
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

  const handleRetry = async (documentId: string) => {
    setRetrying(documentId);
    try {
      await adminApi.retryDocument(documentId);
      toast({
        type: 'success',
        description: 'Document processing restarted'
      });
      // Refresh documents list
      fetchDocuments();
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

  // Modal for document details
  const DocumentDetailsModal = ({ document, isOpen, onClose }: { document: any, isOpen: boolean, onClose: () => void }) => {
    if (!isOpen || !document) return null;
    
    // Format content types for display
    const formatContentTypes = (types: Record<string, number>) => {
      if (!types) return "None";
      return Object.entries(types).map(([type, count]) => (
        `${type}: ${count}`
      )).join(", ");
    };
    
    // Format raptor levels for display
    const formatRaptorLevels = (levels: number[]) => {
      if (!levels || !levels.length) return "None";
      return levels.join(", ");
    };
    
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
        <div className="bg-background p-6 rounded-lg shadow-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold">Document Details</h2>
            <Button variant="ghost" size="sm" onClick={onClose}>×</Button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Document ID</h3>
                <p className="font-mono text-xs break-all">{document.document_id}</p>
              </div>
              
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Filename</h3>
                <p className="break-all">{document.original_filename}</p>
              </div>
              
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Status</h3>
                <Badge className={getStatusBadgeClass(document.status)}>
                  {document.status}
                </Badge>
              </div>
              
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Processing Date</h3>
                <p>{formatDate(document.processing_date)}</p>
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
                <h3 className="text-sm font-medium text-muted-foreground">Images</h3>
                <p>{document.images_count || 0}</p>
              </div>
              
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Content Types</h3>
                <p>{formatContentTypes(document.content_types)}</p>
              </div>
            </div>
          </div>
          
          <div className="mt-4 space-y-2">
            <div>
              <h3 className="text-sm font-medium text-muted-foreground">RAPTOR Levels</h3>
              <p>{formatRaptorLevels(document.raptor_levels)}</p>
            </div>
            
            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Processing Time</h3>
              <p>{document.total_processing_time ? `${document.total_processing_time.toFixed(2)}s` : 'N/A'}</p>
            </div>
            
            {document.error_message && (
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Error</h3>
                <p className="text-red-500">{document.error_message}</p>
              </div>
            )}
          </div>
          
          <div className="mt-6 flex justify-end">
            <Button variant="outline" onClick={onClose}>Close</Button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="container mx-auto py-8 px-4 max-w-7xl">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold">Admin Dashboard</h1>
        <Button 
          onClick={() => {
            fetchStats();
            fetchDocuments(activeTab === 'all' ? undefined : activeTab);
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
                    <label className="block mt-2">
                      <input
                        type="file"
                        className="sr-only"
                        onChange={(e) => {
                          if (e.target.files && e.target.files[0]) {
                            setUploadFile(e.target.files[0]);
                          }
                        }}
                      />
                      <Button className="mt-2">Select File</Button>
                    </label>
                    <p className="text-sm text-muted-foreground mt-2">
                      Supported formats: PDF, DOCX, TXT
                    </p>
                  </>
                )}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
      
      {/* Documents List Section */}
      <Card>
        <CardHeader>
          <CardTitle>Document Management</CardTitle>
          <CardDescription>View and manage document processing</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="all" value={activeTab} onValueChange={handleTabChange}>
            <TabsList className="mb-4">
              <TabsTrigger value="all">
                All
                {statusCounts.processed !== undefined && (
                  <span className="ml-1 text-xs">({documents.length})</span>
                )}
              </TabsTrigger>
              <TabsTrigger value="processed">
                Processed
                {statusCounts.processed !== undefined && (
                  <span className="ml-1 text-xs">({statusCounts.processed})</span>
                )}
              </TabsTrigger>
              <TabsTrigger value="processing">
                Processing
                {statusCounts.processing !== undefined && (
                  <span className="ml-1 text-xs">({statusCounts.processing})</span>
                )}
              </TabsTrigger>
              <TabsTrigger value="failed">
                Failed
                {statusCounts.failed !== undefined && (
                  <span className="ml-1 text-xs">({statusCounts.failed})</span>
                )}
              </TabsTrigger>
              <TabsTrigger value="pending">
                Pending
                {statusCounts.pending !== undefined && (
                  <span className="ml-1 text-xs">({statusCounts.pending})</span>
                )}
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value={activeTab}>
              {loading ? (
                <div className="flex justify-center items-center py-8">
                  <div className="animate-spin mr-2 h-5 w-5 border-t-2 border-b-2 border-primary rounded-full"></div>
                  <span>Loading documents...</span>
                </div>
              ) : documents.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  No documents found
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Document ID</TableHead>
                        <TableHead>Filename</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Processing Date</TableHead>
                        <TableHead>Chunks Count</TableHead>
                        <TableHead>Pages</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {documents.map((doc) => (
                        <TableRow key={doc.document_id}>
                          <TableCell className="font-mono text-xs">
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
                                onClick={(e) => {
                                  // Add debugging
                                  console.log(`Opening document: ${doc.document_id}`);
                                }}
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
                                <span className="truncate max-w-[200px]" title={doc.original_filename}>
                                  {doc.original_filename || '(Unnamed)'}
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
                            {formatDate(doc.processing_date)}
                          </TableCell>
                          <TableCell>
                            {doc.chunks_count || 0}
                          </TableCell>
                          <TableCell>
                            {doc.page_count || 0}
                          </TableCell>
                          <TableCell>
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
                                    <svg
                                      xmlns="http://www.w3.org/2000/svg"
                                      className="h-3 w-3 mr-1"
                                      fill="none"
                                      viewBox="0 0 24 24"
                                      stroke="currentColor"
                                    >
                                      <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                                      />
                                    </svg>
                                    Retry
                                  </>
                                )}
                              </Button>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}