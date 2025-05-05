// app/providers.tsx
'use client';

import { ChatProvider } from '@/contexts/ChatContext';
import { ThemeProvider } from '@/components/theme-provider';
import { SidebarProvider } from '@/components/ui/sidebar';

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
    >
      <ChatProvider>
        <SidebarProvider defaultOpen={true}>
          {children}
        </SidebarProvider>
      </ChatProvider>
    </ThemeProvider>
  );
}