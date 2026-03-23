import type { ChatMessage as ChatMessageType } from "../lib/types";

interface ChatMessageProps {
  message: ChatMessageType;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-lg bg-samhati-600/30 flex items-center justify-center flex-shrink-0 mt-1">
          <span className="text-samhati-300 text-sm font-bold">S</span>
        </div>
      )}

      <div
        className={`max-w-[70%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-samhati-600 text-white"
            : "bg-gray-800 text-gray-100"
        }`}
      >
        <div className="text-sm leading-relaxed whitespace-pre-wrap break-words">
          {renderContent(message.content)}
        </div>

        <div className="flex items-center gap-2 mt-2">
          <span className="text-[10px] opacity-50">
            {message.timestamp.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </span>
          {message.mode && (
            <span className="text-[10px] opacity-50 uppercase">
              {message.mode === "quick" ? "N=3" : "N=7"}
            </span>
          )}
          {message.model && (
            <span className="text-[10px] opacity-50">{message.model}</span>
          )}
        </div>
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-lg bg-gray-700 flex items-center justify-center flex-shrink-0 mt-1">
          <span className="text-gray-300 text-sm font-bold">U</span>
        </div>
      )}
    </div>
  );
}

function renderContent(content: string): React.ReactNode {
  // Simple markdown-like bold rendering
  const parts = content.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={i} className="font-semibold">
          {part.slice(2, -2)}
        </strong>
      );
    }
    return part;
  });
}
