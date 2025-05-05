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