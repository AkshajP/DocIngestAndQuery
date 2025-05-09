// app/page.tsx
'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useChat } from '@/contexts/ChatContext';
import { documentApi } from '@/lib/api';
import { DocumentMetadata } from '@/types/documents';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Separator } from '@/components/ui/separator';
import { FileIcon, LoaderIcon, PlusIcon } from '@/components/icons';
import { toast } from '@/components/toast';
import DocumentSelector from '@/components/ag-grid-document-selector';

export default function Home() {
  const router = useRouter();
  const { chats, loadingChats, fetchChats, createChat } = useChat();
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [loadingDocuments, setLoadingDocuments] = useState(true);
  const [creatingChat, setCreatingChat] = useState(false);

  useEffect(() => {
    fetchChats();
  }, [fetchChats]);

  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        setLoadingDocuments(true);
        const response = await documentApi.listDocuments();
        setDocuments(response.documents);
      } catch (error) {
        console.error('Failed to fetch documents:', error);
        toast({
          type: 'error',
          description: 'Failed to load documents. Please try again.'
        });
      } finally {
        setLoadingDocuments(false);
      }
    };

    fetchDocuments();
  }, []);

  const handleDocumentToggle = (documentId: string) => {
    setSelectedDocuments(prev => {
      if (prev.includes(documentId)) {
        return prev.filter(id => id !== documentId);
      } else {
        return [...prev, documentId];
      }
    });
  };

  const handleCreateChat = async (documentIds: string[]) => {
    if (documentIds.length === 0) {
      toast({
        type: 'error',
        description: 'Please select at least one document to start a chat.'
      });
      return;
    }
  
    setCreatingChat(true);
    try {
      const chatId = await createChat(documentIds);
      router.push(`/chat/${chatId}`);
    } catch (error) {
      console.error('Failed to create chat:', error);
      toast({
        type: 'error',
        description: 'Failed to create chat. Please try again.'
      });
    } finally {
      setCreatingChat(false);
    }
  };

  const processedDocuments = documents.filter(doc => doc.status === 'processed');
  console.log(processedDocuments)

  return (
    <div className="container max-w-6xl mx-auto py-8 px-4 md:px-6">
      <div className="flex flex-col space-y-4 md:space-y-8">
        <div className="flex flex-col space-y-2">
          <h1 className="text-3xl font-bold">Document AI Chat</h1>
          <p className="text-muted-foreground">
            Ask questions about your documents and get AI-powered answers
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
  <CardHeader>
    <CardTitle>Start a New Chat</CardTitle>
    <CardDescription>
      Select documents from the tree view and create a new chat to ask questions
    </CardDescription>
  </CardHeader>
  <CardContent>
    <DocumentSelector 
      onDocumentsSelected={handleCreateChat}
      isLoading={creatingChat}
    />
  </CardContent>
</Card>

          <Card>
            <CardHeader>
              <CardTitle>Recent Chats</CardTitle>
              <CardDescription>
                Continue a previous conversation
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loadingChats ? (
                <div className="flex justify-center items-center p-6">
                  <LoaderIcon className="animate-spin mr-2" />
                  <span>Loading chats...</span>
                </div>
              ) : chats.length === 0 ? (
                <div className="p-6 text-center">
                  <p className="text-muted-foreground">No chat history found</p>
                  <p className="text-sm mt-2">Select documents and start a new chat</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {chats.slice(0, 5).map(chat => (
                    <div
                      key={chat.id}
                      className="p-3 border rounded-md hover:bg-accent/50 cursor-pointer"
                      onClick={() => router.push(`/chat/${chat.id}`)}
                    >
                      <div className="font-medium truncate">{chat.title}</div>
                      <div className="flex justify-between items-center mt-1">
                        <span className="text-xs text-muted-foreground">
                          {chat.messages_count} message{chat.messages_count !== 1 && 's'}
                        </span>
                        {chat.last_active && (
                          <span className="text-xs text-muted-foreground">
                            {new Date(chat.last_active).toLocaleDateString()}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
            {chats.length > 5 && (
              <CardFooter>
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={() => router.push('/chats')}
                >
                  View All Chats
                </Button>
              </CardFooter>
            )}
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>How It Works</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="flex flex-col items-center text-center p-4">
                <div className="bg-primary/10 p-3 rounded-full mb-4">
                  <FileIcon className="h-6 w-6 text-primary" />
                </div>
                <h3 className="font-medium mb-2">1. Select Documents</h3>
                <p className="text-sm text-muted-foreground">
                  Choose which documents you want to query from your document repository
                </p>
              </div>

              <div className="flex flex-col items-center text-center p-4">
                <div className="bg-primary/10 p-3 rounded-full mb-4">
                  <PlusIcon className="h-6 w-6 text-primary" />
                </div>
                <h3 className="font-medium mb-2">2. Create a Chat</h3>
                <p className="text-sm text-muted-foreground">
                  Start a new conversation with your selected documents as context
                </p>
              </div>

              <div className="flex flex-col items-center text-center p-4">
                <div className="bg-primary/10 p-3 rounded-full mb-4">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6 text-primary"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                    />
                  </svg>
                </div>
                <h3 className="font-medium mb-2">3. Ask Questions</h3>
                <p className="text-sm text-muted-foreground">
                  Get AI-powered answers based on the content of your documents
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}