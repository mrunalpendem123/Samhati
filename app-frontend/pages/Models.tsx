import { useState, useEffect, useCallback } from "react";
import { useNode } from "../hooks/useNode";
import ModelCard from "../components/ModelCard";
import type { ModelInfo } from "../lib/types";
import * as api from "../lib/api";

export default function Models() {
  const { status, start } = useNode();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);

  const fetchModels = useCallback(async () => {
    try {
      const list = await api.listModels();
      setModels(list);
    } catch {
      // Failed to fetch models
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const handleDownload = async (modelId: string) => {
    setDownloadingId(modelId);
    try {
      await api.downloadModel(modelId);
      await fetchModels();
    } catch {
      // Download failed
    } finally {
      setDownloadingId(null);
    }
  };

  const handleSelect = async (modelId: string) => {
    const model = models.find((m) => m.id === modelId);
    if (model) {
      await start(model.id);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white">Models</h2>
        <p className="text-sm text-gray-500">
          Download specialist models to earn bonus SMTI rewards
        </p>
      </div>

      {/* Currently running */}
      {status.model && (
        <div className="glass-panel p-4 flex items-center gap-3 border-samhati-600/30">
          <div className="w-3 h-3 rounded-full bg-green-400 animate-pulse" />
          <span className="text-sm text-gray-300">
            Currently running:{" "}
            <span className="text-white font-medium">{status.model}</span>
          </span>
        </div>
      )}

      {/* Model grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {models.map((model) => (
          <ModelCard
            key={model.id}
            model={model}
            isRunning={status.model === model.id}
            onDownload={handleDownload}
            onSelect={handleSelect}
            downloading={downloadingId === model.id}
          />
        ))}
      </div>

      {models.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500">Loading model catalogue...</p>
        </div>
      )}
    </div>
  );
}
