import type { ModelInfo } from "../lib/types";

interface ModelCardProps {
  model: ModelInfo;
  isRunning: boolean;
  onDownload: (id: string) => void;
  onSelect: (id: string) => void;
  downloading: boolean;
}

export default function ModelCard({
  model,
  isRunning,
  onDownload,
  onSelect,
  downloading,
}: ModelCardProps) {
  const domainColors: Record<string, string> = {
    General: "bg-blue-500/20 text-blue-300",
    Hindi: "bg-orange-500/20 text-orange-300",
    Code: "bg-green-500/20 text-green-300",
    Medical: "bg-red-500/20 text-red-300",
    Math: "bg-purple-500/20 text-purple-300",
  };

  const badgeColor = domainColors[model.domain] || "bg-gray-500/20 text-gray-300";

  return (
    <div className={`glass-panel p-5 flex flex-col gap-3 ${isRunning ? "ring-2 ring-samhati-500/50" : ""}`}>
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-base font-semibold text-white">{model.name}</h3>
          <span className={`inline-block mt-1 px-2 py-0.5 rounded-full text-xs font-medium ${badgeColor}`}>
            {model.domain}
          </span>
        </div>
        <span className="text-xs text-gray-500 font-mono">{model.size_display}</span>
      </div>

      {/* Description */}
      <p className="text-sm text-gray-400 leading-relaxed">{model.description}</p>

      {/* SMTI Bonus */}
      <div className="flex items-center gap-1.5">
        <svg className="w-3.5 h-3.5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
          <path d="M10.868 2.884c-.321-.772-1.415-.772-1.736 0l-1.83 4.401-4.753.381c-.833.067-1.171 1.107-.536 1.651l3.62 3.102-1.106 4.637c-.194.813.691 1.456 1.405 1.02L10 15.591l4.069 2.485c.713.436 1.598-.207 1.404-1.02l-1.106-4.637 3.62-3.102c.635-.544.297-1.584-.536-1.65l-4.752-.382-1.831-4.401Z" />
        </svg>
        <span className="text-xs text-yellow-400/80">{model.smti_bonus}</span>
      </div>

      {/* Progress bar */}
      {model.download_progress != null && (
        <div className="w-full bg-gray-800 rounded-full h-2">
          <div
            className="bg-samhati-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${model.download_progress}%` }}
          />
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2 mt-auto">
        {!model.downloaded ? (
          <button
            onClick={() => onDownload(model.id)}
            disabled={downloading}
            className="btn-primary text-sm flex-1"
          >
            {downloading ? "Downloading..." : "Download"}
          </button>
        ) : isRunning ? (
          <span className="flex-1 text-center py-2 text-sm font-medium text-green-400">
            Currently Running
          </span>
        ) : (
          <button
            onClick={() => onSelect(model.id)}
            className="btn-secondary text-sm flex-1"
          >
            Use This Model
          </button>
        )}
      </div>
    </div>
  );
}
