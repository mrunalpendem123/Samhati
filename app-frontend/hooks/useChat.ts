import { useState, useCallback } from "react";
import type { ChatMessage, ChatMode } from "../lib/types";
import * as api from "../lib/api";

let messageCounter = 0;
function genId(): string {
  return `msg_${Date.now()}_${++messageCounter}`;
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [mode, setMode] = useState<ChatMode>("quick");
  const [sending, setSending] = useState(false);

  const send = useCallback(
    async (content: string) => {
      if (!content.trim() || sending) return;

      const userMsg: ChatMessage = {
        id: genId(),
        role: "user",
        content: content.trim(),
        timestamp: new Date(),
        mode,
      };

      setMessages((prev) => [...prev, userMsg]);
      setSending(true);

      try {
        const response = await api.sendChat(content.trim(), mode);
        const assistantMsg: ChatMessage = {
          id: genId(),
          role: "assistant",
          content: response,
          timestamp: new Date(),
          mode,
          model: "samhati-mesh",
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (e) {
        const errorMsg: ChatMessage = {
          id: genId(),
          role: "assistant",
          content: `Error: ${String(e)}`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMsg]);
      } finally {
        setSending(false);
      }
    },
    [mode, sending]
  );

  const clear = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, mode, setMode, sending, send, clear };
}
