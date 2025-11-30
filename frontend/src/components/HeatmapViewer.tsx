import React, { useState } from "react";
import { Eye, Layers, Zap } from "lucide-react";

interface HeatmapViewerProps {
  originalUrl?: string;
  elaUrl?: string;
  gradCamUrl?: string;
}

type ViewMode = "original" | "ela" | "gradcam";

export const HeatmapViewer: React.FC<HeatmapViewerProps> = ({
  originalUrl,
  elaUrl,
  gradCamUrl,
}) => {
  const [mode, setMode] = useState<ViewMode>("original");

  const currentImage = (): string | undefined => {
    switch (mode) {
      case "ela":
        return elaUrl;
      case "gradcam":
        return gradCamUrl;
      default:
        return originalUrl;
    }
  };

  const src = currentImage();

  return (
    <div className="flex flex-col h-full bg-slate-800 rounded-xl overflow-hidden border border-slate-700 shadow-lg">
      <div className="p-4 border-b border-slate-700 flex justify-between items-center bg-slate-800/50 backdrop-blur-sm">
        <h3 className="font-semibold text-slate-200">Visual Analysis</h3>
        <div className="flex bg-slate-900 rounded-lg p-1 gap-1">
          <button
            onClick={() => setMode("original")}
            className={`px-3 py-1.5 rounded-md text-xs font-medium flex items-center gap-2 transition-colors ${
              mode === "original"
                ? "bg-cyan-600 text-white"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
            }`}
          >
            <Eye size={14} /> Original
          </button>
          <button
            onClick={() => setMode("ela")}
            className={`px-3 py-1.5 rounded-md text-xs font-medium flex items-center gap-2 transition-colors ${
              mode === "ela"
                ? "bg-cyan-600 text-white"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
            }`}
          >
            <Layers size={14} /> ELA
          </button>
          <button
            onClick={() => setMode("gradcam")}
            className={`px-3 py-1.5 rounded-md text-xs font-medium flex items-center gap-2 transition-colors ${
              mode === "gradcam"
                ? "bg-cyan-600 text-white"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
            }`}
          >
            <Zap size={14} /> GradCAM
          </button>
        </div>
      </div>

      <div className="relative grow bg-slate-950 flex items-center justify-center min-h-[400px]">
        <div
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: "radial-gradient(#475569 1px, transparent 1px)",
            backgroundSize: "20px 20px",
          }}
        ></div>

        {src ? (
          <img
            src={src}
            alt={`Analysis View - ${mode}`}
            className="max-h-[500px] w-auto max-w-full object-contain p-4 transition-opacity duration-300"
          />
        ) : (
          <div className="text-slate-500 text-sm">
            No image available for {mode.toUpperCase()} view.
          </div>
        )}

        <div className="absolute bottom-4 right-4 px-3 py-1 bg-black/70 backdrop-blur text-white text-xs rounded-full border border-white/10">
          Showing: {mode.toUpperCase()}
        </div>
      </div>
    </div>
  );
};
