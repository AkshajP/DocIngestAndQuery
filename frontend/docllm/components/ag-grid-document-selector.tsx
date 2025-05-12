// frontend/docllm/components/ag-grid-document-selector.tsx
'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { AgGridReact } from 'ag-grid-react';

// Import required modules
import { ModuleRegistry } from 'ag-grid-community'; 
import { AllEnterpriseModule } from 'ag-grid-enterprise';

// Register all Community and Enterprise features
ModuleRegistry.registerModules([AllEnterpriseModule]);

// Import styles
//import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';

import { documentApi } from '@/lib/api';
import { transformDocumentsToTree } from '@/lib/document-tree-util';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { LoaderIcon, FileIcon, FolderIcon } from 'lucide-react';
interface DocumentSelectorProps {
  onDocumentsSelected: (documentIds: string[]) => void;
  isLoading?: boolean;
}

const DocumentSelector: React.FC<DocumentSelectorProps> = ({
  onDocumentsSelected,
  isLoading = false
}) => {
  const [rowData, setRowData] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  
  // Column definitions
  const columnDefs = useMemo(() => [
    {
      headerName: 'Selection',
      width: 120,
      field: 'docId',
      filter: false,
      cellRenderer: (params: any) => {
        if (params.data.isFolder) return null;
        const isDisabled = ['failed', 'processing'].includes(params.data.status);
        return (
          <Checkbox
            disabled={isDisabled}
            checked={selectedDocuments.includes(params.value)}
            onCheckedChange={(checked) => {
              if (checked) {
                setSelectedDocuments(prev => [...prev, params.value]);
              } else {
                setSelectedDocuments(prev => prev.filter(id => id !== params.value));
              }
            }}
          />
        );
      }
    },
    {
      field: 'name',
      headerName: 'Document Name',
      cellRenderer: 'agGroupCellRenderer',
      cellRendererParams: {
        innerRenderer: (params: any) => {
          const isFolder = params.data.isFolder;
          return (
            <div className="flex items-center">
              {isFolder ? (
                <FolderIcon className="mr-2 h-4 w-4 text-yellow-500" />
              ) : (
                <FileIcon className="mr-2 h-4 w-4 text-blue-500" />
              )}
              <span>{params.value}</span>
            </div>
          );
        }
      },
      
    },
    {
      field: 'description',
      headerName: 'Description',
      filter: true,
      resizable: true,
    },
    {
      field: 'status',
      headerName: 'Status',
      width: 120,
      filter: true,
      resizable: false,
      cellRenderer: (params: any) => {
        if (!params.value || params.data.isFolder) return null;
        
        const getStatusClass = (status: string) => {
          switch (status) {
            case 'processed': return 'bg-green-100 text-green-800';
            case 'processing': return 'bg-blue-100 text-blue-800';
            case 'failed': return 'bg-red-100 text-red-800';
            case 'pending': return 'bg-yellow-100 text-yellow-800';
            default: return 'bg-gray-100 text-gray-800';
          }
        };
        
        return (
          <Badge className={getStatusClass(params.value)}>
            {params.value}
          </Badge>
        );
      }
    },
    {
      field: 'importance',
      headerName: 'Importance',
      resizable: false,
      filter: 'agNumberColumnFilter',
      cellRenderer: (params: any) => {
        if (!params.value || params.data.isFolder) return null;
        
        // Show stars based on importance
        const stars = Array(params.value).fill('★').join('');
        return <span className="text-yellow-500">{stars}</span>;
      }
    },
    {
      field: 'page_count',
      headerName: 'Pages',
      filter: 'agNumberColumnFilter'
    },
    {
      field: 'size',
      headerName: 'Size',
    },
    
  ], [selectedDocuments]);
  
  // Grid options
  const defaultColDef = useMemo(() => ({
    sortable: true,
    resizable: true,
    filter: true
  }), []);
  
  // Fetch documents and transform into tree
  const fetchDocuments = useCallback(async () => {
    try {
      setLoading(true);
      const response = await documentApi.listDocuments();
      const documents = response.documents;
      
      // Transform flat list to tree structure
      const treeData = transformDocumentsToTree(documents);
      setRowData(treeData);
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    } finally {
      setLoading(false);
    }
  }, []);
  
  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);
  
  // Handle selection
  const handleSelectionConfirm = () => {
    onDocumentsSelected(selectedDocuments);
  };
  
  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-medium">Select Documents</h3>
        <div className="flex gap-2">
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => setSelectedDocuments([])}
            disabled={selectedDocuments.length === 0}
          >
            Clear Selection
          </Button>
          <Button
            size="sm"
            onClick={handleSelectionConfirm}
            disabled={selectedDocuments.length === 0 || isLoading}
          >
            {isLoading ? (
              <>
                <LoaderIcon className="mr-2 h-4 w-4 animate-spin" />
                Creating Chat...
              </>
            ) : (
              `Start Chat with ${selectedDocuments.length} Document${selectedDocuments.length !== 1 ? 's' : ''}`
            )}
          </Button>
        </div>
      </div>
      
      {loading ? (
        <div className="flex justify-center items-center p-6">
          <LoaderIcon className="animate-spin mr-2" />
          <span>Loading documents...</span>
        </div>
      ) : (
        <div className="ag-theme-alpine h-[500px] w-full">
          <AgGridReact
            rowData={rowData}
            columnDefs={columnDefs}
            defaultColDef={defaultColDef}
            animateRows={true}
            treeData={true}
            groupDefaultExpanded={1}
            theme="legacy" // Add this line to fix the theming error
            getDataPath={(data) => {
              return data.isFolder ? [data.name] : [data.path.split('/')[0], data.name];
            }}
            autoGroupColumnDef={{
              headerName: 'Document Hierarchy',
              minWidth: 300,
              cellRendererParams: {
                suppressCount: true
              }
            }}
          />
        </div>
      )}
      
      <div className="mt-4">
        <p className="text-sm text-muted-foreground">
          {selectedDocuments.length} document{selectedDocuments.length !== 1 ? 's' : ''} selected
        </p>
      </div>
    </div>
  );
};

export default DocumentSelector;