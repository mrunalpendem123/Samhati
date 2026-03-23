import { useState } from "react";
import type { AppSettings } from "../lib/types";

const defaultSettings: AppSettings = {
  modelsDir: "~/.samhati/models",
  maxVramMb: 8192,
  maxRamMb: 16384,
  maxConnections: 64,
  relayPreference: "auto",
  autoUpdateModels: true,
  theme: "dark",
};

export default function Settings() {
  const [settings, setSettings] = useState<AppSettings>(defaultSettings);
  const [saved, setSaved] = useState(false);

  const update = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
    setSaved(false);
  };

  const handleSave = () => {
    // TODO: Persist settings via Tauri command
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  return (
    <div className="p-6 space-y-6 max-w-2xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Settings</h2>
          <p className="text-sm text-gray-500">Configure your Samhati node</p>
        </div>
        <button onClick={handleSave} className="btn-primary">
          {saved ? "Saved!" : "Save Settings"}
        </button>
      </div>

      {/* Storage */}
      <section className="glass-panel p-5 space-y-4">
        <h3 className="text-sm font-semibold text-gray-300">Storage</h3>
        <div>
          <label className="text-xs text-gray-400 block mb-1">
            Model Storage Directory
          </label>
          <input
            type="text"
            value={settings.modelsDir}
            onChange={(e) => update("modelsDir", e.target.value)}
            className="input-field font-mono text-sm"
          />
        </div>
      </section>

      {/* Resources */}
      <section className="glass-panel p-5 space-y-4">
        <h3 className="text-sm font-semibold text-gray-300">
          Resource Allocation
        </h3>
        <div>
          <label className="text-xs text-gray-400 block mb-1">
            Max VRAM (MB)
          </label>
          <input
            type="number"
            value={settings.maxVramMb}
            onChange={(e) => update("maxVramMb", parseInt(e.target.value) || 0)}
            className="input-field font-mono text-sm"
            min={512}
            max={49152}
            step={512}
          />
          <p className="text-xs text-gray-600 mt-1">
            GPU memory allocated for inference. Higher = larger models.
          </p>
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">
            Max RAM (MB)
          </label>
          <input
            type="number"
            value={settings.maxRamMb}
            onChange={(e) => update("maxRamMb", parseInt(e.target.value) || 0)}
            className="input-field font-mono text-sm"
            min={1024}
            max={131072}
            step={1024}
          />
          <p className="text-xs text-gray-600 mt-1">
            System memory for KV cache and model offloading.
          </p>
        </div>
      </section>

      {/* Network */}
      <section className="glass-panel p-5 space-y-4">
        <h3 className="text-sm font-semibold text-gray-300">Network</h3>
        <div>
          <label className="text-xs text-gray-400 block mb-1">
            Max Connections
          </label>
          <input
            type="number"
            value={settings.maxConnections}
            onChange={(e) =>
              update("maxConnections", parseInt(e.target.value) || 1)
            }
            className="input-field font-mono text-sm"
            min={1}
            max={256}
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">
            Relay Preference
          </label>
          <select
            value={settings.relayPreference}
            onChange={(e) =>
              update(
                "relayPreference",
                e.target.value as AppSettings["relayPreference"]
              )
            }
            className="input-field text-sm"
          >
            <option value="auto">Auto (recommended)</option>
            <option value="direct">Direct only</option>
            <option value="relay">Relay only</option>
          </select>
          <p className="text-xs text-gray-600 mt-1">
            "Auto" uses direct QUIC connections when possible, falling back to
            relay nodes behind NAT.
          </p>
        </div>
      </section>

      {/* Model updates */}
      <section className="glass-panel p-5 space-y-4">
        <h3 className="text-sm font-semibold text-gray-300">Updates</h3>
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={settings.autoUpdateModels}
            onChange={(e) => update("autoUpdateModels", e.target.checked)}
            className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-samhati-500 focus:ring-samhati-500"
          />
          <div>
            <span className="text-sm text-gray-200">Auto-update models</span>
            <p className="text-xs text-gray-600">
              Automatically download updated model weights when available.
            </p>
          </div>
        </label>
      </section>

      {/* Appearance */}
      <section className="glass-panel p-5 space-y-4">
        <h3 className="text-sm font-semibold text-gray-300">Appearance</h3>
        <div>
          <label className="text-xs text-gray-400 block mb-1">Theme</label>
          <select
            value={settings.theme}
            onChange={(e) =>
              update("theme", e.target.value as AppSettings["theme"])
            }
            className="input-field text-sm"
          >
            <option value="dark">Dark</option>
            <option value="light">Light</option>
          </select>
        </div>
      </section>
    </div>
  );
}
