import { useNode } from "../hooks/useNode";

export default function NodeStatus() {
  const { status } = useNode();

  return (
    <div className="flex items-center gap-2">
      <div
        className={`w-2.5 h-2.5 rounded-full ${
          status.running
            ? "bg-green-400 shadow-lg shadow-green-400/50 animate-pulse"
            : "bg-gray-600"
        }`}
      />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-200 truncate">
          {status.running ? "Node Online" : "Node Offline"}
        </p>
        {status.running && status.model && (
          <p className="text-xs text-gray-500 truncate">{status.model}</p>
        )}
      </div>
      <span className="text-xs font-mono text-samhati-400">
        {status.elo_score}
      </span>
    </div>
  );
}
