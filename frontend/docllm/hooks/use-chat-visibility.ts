// frontend/docllm/hooks/use-chat-visibility.ts
import { useState } from 'react';

export function useChatVisibility() {
  const [visibleChats, setVisibleChats] = useState<Record<string, boolean>>({});

  const toggleChatVisibility = (chatId: string) => {
    setVisibleChats((prev) => ({
      ...prev,
      [chatId]: !prev[chatId]
    }));
  };

  const isChatVisible = (chatId: string) => {
    return visibleChats[chatId] || false;
  };

  return {
    visibleChats,
    toggleChatVisibility,
    isChatVisible
  };
}