// app/(main)/layout.tsx
'use client';

import { useEffect } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarFooter,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from '@/components/ui/sidebar';
import { Button } from '@/components/ui/button';
import { DocumentSidebar } from '@/components/document-sidebar';
import { useChat } from '@/contexts/ChatContext';
import { ChatItem } from '@/components/sidebar-history-item';
import { PlusIcon } from '@/components/icons';
import { chatApi } from '@/lib/api';

export default function MainLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { chats, loadingChats, fetchChats, createChat } = useChat();
  const router = useRouter();
  const pathname = usePathname();
  
  useEffect(() => {
    fetchChats();
  }, [fetchChats]);
  
  const handleCreateChat = async (documentIds: string[]) => {
    try {
      const chatId = await createChat(documentIds);
      router.push(`/chat/${chatId}`);
    } catch (error) {
      console.error('Failed to create chat:', error);
    }
  };
  
  const getCurrentChatId = (): string | null => {
    const match = pathname.match(/\/chat\/([^\/]+)/);
    return match ? match[1] : null;
  };
  
  const currentChatId = getCurrentChatId();

  return (
    <div className="flex h-screen">
      <Sidebar>
        <SidebarHeader>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton asChild>
                <Button
                  className="w-full justify-start"
                  onClick={() => router.push('/')}
                >
                  <PlusIcon className="mr-2" />
                  New Chat
                </Button>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarHeader>
        
        <SidebarContent>
          {pathname === '/' ? (
            <DocumentSidebar onDocumentsSelected={handleCreateChat} />
          ) : currentChatId ? (
            <DocumentSidebar chatId={currentChatId} />
          ) : null}
          
          {chats.length > 0 && (
            <div className="mt-4">
              <h3 className="px-4 mb-2 text-sm font-semibold">Recent Chats</h3>
              <SidebarMenu>
                {chats.map(chat => (
                  <ChatItem
                    key={chat.id}
                    chat={chat}
                    isActive={currentChatId === chat.id}
                    onDelete={async (chatId) => {
                      try {
                        await chatApi.deleteChat(chatId);
                        fetchChats();
                        if (currentChatId === chatId) {
                          router.push('/');
                        }
                      } catch (error) {
                        console.error('Failed to delete chat:', error);
                      }
                    }}
                    setOpenMobile={() => {}}
                  />
                ))}
              </SidebarMenu>
            </div>
          )}
        </SidebarContent>
        
          <SidebarFooter>
      <div className="px-4 py-2">
        <Button 
          variant="ghost" 
          className="w-full justify-start text-muted-foreground" 
          onClick={() => router.push('/admin')}
        >
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            className="h-4 w-4 mr-2" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" 
            />
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" 
            />
          </svg>
          Admin Dashboard
        </Button>
      </div>
      <div className="p-4 text-xs text-muted-foreground">
        Document AI Chat System
      </div>
  </SidebarFooter>
      </Sidebar>
      
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  );
}