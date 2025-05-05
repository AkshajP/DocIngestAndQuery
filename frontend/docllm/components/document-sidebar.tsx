// components/document-sidebar.tsx
'use client';

import React, { useState, useEffect } from 'react';
import { documentApi, chatApi } from '@/lib/api';
import { DocumentMetadata, DocumentStatus } from '@/types/documents';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
} from '@/components/ui/sidebar';
import { FileIcon, LoaderIcon } from '@/components/icons';
import { toast } from '@/components/toast';

interface DocumentSidebarProps {
  chatId?: string;
  onDocumentsSelected?: (documentIds: string[]) => void;
}

export function DocumentSidebar({ chatId, onDocumentsSelected }: DocumentSidebarProps) {
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadedDocumentIds, setLoadedDocumentIds] = useState<string[]>([]);

  // Fetch documents
  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        setLoading(true);
        const response = await documentApi.listDocuments();
        setDocuments(response.documents);
      } catch (error) {
        toast({
          type: 'error',
          description: 'Failed to load documents. Please try again.'
        });
      } finally {
        setLoading(false);
      }
    };

    fetchDocuments();
  }, []);

  // If chatId is provided, fetch loaded documents for this chat
  useEffect(() => {
    if (!chatId) return;

    const fetchChatDocuments = async () => {
      try {
        const chatDetails = await chatApi.getChat(chatId);
        const loadedDocs = chatDetails.loaded_documents.map(doc => doc.document_id);
        setLoadedDocumentIds(loadedDocs);
        setSelectedDocuments(loadedDocs);
      } catch (error) {
        console.error("Failed to fetch chat documents:", error);
      }
    };

    fetchChatDocuments();
  }, [chatId]);

  const handleDocumentToggle = (documentId: string) => {
    setSelectedDocuments(prev => {
      if (prev.includes(documentId)) {
        return prev.filter(id => id !== documentId);
      } else {
        return [...prev, documentId];
      }
    });
  };

  const handleUpdateChatDocuments = async () => {
    if (!chatId) return;
    
    try {
      // Determine which docs to add and which to remove
      const docsToAdd = selectedDocuments.filter(id => !loadedDocumentIds.includes(id));
      const docsToRemove = loadedDocumentIds.filter(id => !selectedDocuments.includes(id));
      
      await chatApi.updateChatDocuments(chatId, {
        add: docsToAdd,
        remove: docsToRemove,
      });
      
      // Update loadedDocumentIds to reflect the changes
      setLoadedDocumentIds(selectedDocuments);
      
      toast({
        type: 'success',
        description: 'Documents updated successfully.'
      });
    } catch (error) {
      toast({
        type: 'error',
        description: 'Failed to update documents. Please try again.'
      });
    }
  };

  const handleApplySelection = () => {
    if (onDocumentsSelected) {
      onDocumentsSelected(selectedDocuments);
    }
  };

  return (
    <SidebarGroup>
      <SidebarGroupLabel>Documents</SidebarGroupLabel>
      <SidebarGroupContent>
        {loading ? (
          <div className="flex items-center justify-center p-4">
            <LoaderIcon className="animate-spin mr-2" />
            <span>Loading documents...</span>
          </div>
        ) : documents.length === 0 ? (
          <div className="p-4 text-muted-foreground">
            No documents available. Contact administrator to add documents.
          </div>
        ) : (
          <div className="space-y-4">
            <div className="space-y-2">
              {documents.map(doc => (
                <div
                  key={doc.document_id}
                  className={`flex items-center p-2 rounded-md ${
                    doc.status !== DocumentStatus.PROCESSED ? 'opacity-60' : ''
                  }`}
                >
                  <Checkbox
                    id={`doc-${doc.document_id}`}
                    checked={selectedDocuments.includes(doc.document_id)}
                    onCheckedChange={() => {
                      if (doc.status === DocumentStatus.PROCESSED) {
                        handleDocumentToggle(doc.document_id);
                      } else {
                        toast({
                          type: 'error',
                          description: 'Document is not fully processed yet.'
                        });
                      }
                    }}
                    disabled={doc.status !== DocumentStatus.PROCESSED}
                    className="mr-2"
                  />
                  <label
                    htmlFor={`doc-${doc.document_id}`}
                    className={`flex items-center flex-1 cursor-pointer ${
                      doc.status !== DocumentStatus.PROCESSED ? 'cursor-not-allowed' : ''
                    }`}
                  >
                    <FileIcon className="mr-2 flex-shrink-0" />
                    <span className="text-sm truncate" title={doc.original_filename}>
                      {doc.original_filename}
                    </span>
                  </label>
                  <span className={`ml-2 text-xs px-2 py-1 rounded-full ${
                    getStatusBadgeClass(doc.status)
                  }`}>
                    {doc.status}
                  </span>
                </div>
              ))}
            </div>
            
            <div className="pt-2">
              {chatId ? (
                <Button 
                  className="w-full" 
                  onClick={handleUpdateChatDocuments}
                  disabled={selectedDocuments.length === 0 || JSON.stringify(selectedDocuments.sort()) === JSON.stringify(loadedDocumentIds.sort())}
                >
                  Update Chat Documents
                </Button>
              ) : (
                <Button 
                  className="w-full" 
                  onClick={handleApplySelection}
                  disabled={selectedDocuments.length === 0}
                >
                  Create Chat with Selected
                </Button>
              )}
            </div>
          </div>
        )}
      </SidebarGroupContent>
    </SidebarGroup>
  );
}

function getStatusBadgeClass(status: DocumentStatus): string {
  switch (status) {
    case DocumentStatus.PROCESSED:
      return 'bg-green-100 text-green-800';
    case DocumentStatus.PROCESSING:
      return 'bg-blue-100 text-blue-800';
    case DocumentStatus.PENDING:
      return 'bg-yellow-100 text-yellow-800';
    case DocumentStatus.FAILED:
      return 'bg-red-100 text-red-800';
    case DocumentStatus.DELETED:
      return 'bg-gray-100 text-gray-800';
    default:
      return 'bg-gray-100 text-gray-800';
  }
}