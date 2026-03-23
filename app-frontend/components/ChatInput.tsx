import { useState, useRef, useEffect } from "react";
import type { ChatMode } from "../lib/types";

interface ChatInputProps {
  onSend: (message: string) => void;
  mode: ChatMode;
  onModeChange: (mode: ChatMode) => void;
  disabled?: boolean;
}

export default function ChatInput({
  onSend,
  mode,
  onModeChange,
  disabled = false,
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [input]);

  const handleSubmit = () => {
    if (input.trim() && !disabled) {
      onSend(input);
      setInput("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-gray-800 bg-gray-900/80 backdrop-blur-sm p-4">
      {/* Mode toggle */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xs text-gray-500">Mode:</span>
        <button
          onClick={() => onModeChange("quick")}
          className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
            mode === "quick"
              ? "bg-green-600/30 text-green-300 border border-green-600/50"
              : "bg-gray-800 text-gray-500 hover:text-gray-300"
          }`}
        >
          Quick (N=3)
        </button>
        <button
          onClick={() => onModeChange("best")}
          className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
            mode === "best"
              ? "bg-purple-600/30 text-purple-300 border border-purple-600/50"
              : "bg-gray-800 text-gray-500 hover:text-gray-300"
          }`}
        >
          Best (N=7)
        </button>
      </div>

      {/* Input area */}
      <div className="flex items-end gap-3">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          disabled={disabled}
          rows={1}
          className="flex-1 resize-none bg-gray-800 border border-gray-700 rounded-xl px-4 py-3
                     text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2
                     focus:ring-samhati-500 focus:border-transparent transition-all duration-150
                     disabled:opacity-50"
        />
        <button
          onClick={handleSubmit}
          disabled={disabled || !input.trim()}
          className="px-4 py-3 bg-samhati-600 hover:bg-samhati-500 disabled:opacity-30
                     disabled:cursor-not-allowed text-white rounded-xl transition-colors duration-150"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L5.999 12Zm0 0h7.5" />
          </svg>
        </button>
      </div>
    </div>
  );
}
