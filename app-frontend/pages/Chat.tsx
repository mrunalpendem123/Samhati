import { useEffect, useRef } from "react";
import { useChat } from "../hooks/useChat";
import ChatMessage from "../components/ChatMessage";
import ChatInput from "../components/ChatInput";

export default function Chat() {
  const { messages, mode, setMode, sending, send, clear } = useChat();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-gray-900/50">
        <div>
          <h2 className="text-lg font-semibold text-white">Chat</h2>
          <p className="text-xs text-gray-500">
            Powered by the Samhati mesh network
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500">
            {messages.length} messages
          </span>
          {messages.length > 0 && (
            <button onClick={clear} className="btn-secondary text-xs px-3 py-1">
              Clear
            </button>
          )}
        </div>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 rounded-2xl bg-samhati-600/20 flex items-center justify-center mb-4">
              <span className="text-3xl text-samhati-400 font-bold">S</span>
            </div>
            <h3 className="text-xl font-semibold text-gray-300 mb-2">
              Welcome to Samhati
            </h3>
            <p className="text-sm text-gray-500 max-w-md leading-relaxed">
              Your queries are routed to the nearest specialist nodes in the
              decentralized mesh. Choose{" "}
              <span className="text-green-400">Quick (N=3)</span> for speed or{" "}
              <span className="text-purple-400">Best (N=7)</span> for quality.
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}

        {sending && (
          <div className="flex items-center gap-2 text-gray-500">
            <div className="flex gap-1">
              <span className="w-2 h-2 bg-samhati-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
              <span className="w-2 h-2 bg-samhati-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
              <span className="w-2 h-2 bg-samhati-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
            </div>
            <span className="text-xs">Querying mesh network...</span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <ChatInput
        onSend={send}
        mode={mode}
        onModeChange={setMode}
        disabled={sending}
      />
    </div>
  );
}
