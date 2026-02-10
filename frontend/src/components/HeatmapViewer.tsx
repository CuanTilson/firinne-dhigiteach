import React, { useState } from "react";
import { Eye, Layers, Zap, Activity, Aperture } from "lucide-react";

interface HeatmapViewerProps {
  originalUrl?: string;
  elaUrl?: string;
  gradCamUrl?: string;
  noiseUrl?: string;
  jpegQualityUrl?: string;
}

type ViewMode = "original" | "ela" | "gradcam" | "noise" | "jpeg";

export const HeatmapViewer: React.FC<HeatmapViewerProps> = ({
  originalUrl,
  elaUrl,
  gradCamUrl,
  noiseUrl,
  jpegQualityUrl,
}) => {
  const [mode, setMode] = useState<ViewMode>("original");

  const currentImage = (): string | undefined => {
    switch (mode) {
      case "ela":
        return elaUrl;
      case "gradcam":
        return gradCamUrl;
      case "noise":
        return noiseUrl;
      case "jpeg":
        return jpegQualityUrl;
      default:
        return originalUrl;
    }
  };

  const src = currentImage();

  return (
    <div className="fd-card overflow-hidden">
      <div className="p-4 border-b border-slate-800 flex flex-wrap gap-3 items-center justify-between">
        <div>
          <div className="fd-kicker">Visual Evidence</div>
          <h3 className="text-lg font-semibold text-slate-100 fd-title">
            Visual Analysis
          </h3>
        </div>
        <div className="flex flex-wrap gap-2">
          {(
            [
              ["original", Eye, "Original"],
              ["ela", Layers, "ELA"],
              ["gradcam", Zap, "GradCAM"],
              ["noise", Activity, "Noise"],
              ["jpeg", Aperture, "JPEG"],
            ] as const
          ).map(([key, Icon, label]) => (
            <button
              key={key}
              onClick={() => setMode(key)}
              className={`px-3 py-1.5 rounded-full text-xs font-medium flex items-center gap-2 transition-colors border ${
                mode === key
                  ? "bg-cyan-500/20 text-cyan-200 border-cyan-400/40"
                  : "text-slate-400 border-slate-800 hover:text-slate-200 hover:bg-slate-900"
              }`}
            >
              <Icon size={14} /> {label}
            </button>
          ))}
        </div>
      </div>

      <div className="relative bg-slate-950/70 flex items-center justify-center min-h-[420px]">
        <div
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: "radial-gradient(#334155 1px, transparent 1px)",
            backgroundSize: "24px 24px",
          }}
        ></div>

        {src ? (
          <img
            src={src}
            alt={`Analysis View - ${mode}`}
            className="max-h-[520px] w-auto max-w-full object-contain p-6 transition-opacity duration-300"
          />
        ) : (
          <div className="text-slate-500 text-sm">
            No image available for {mode.toUpperCase()} view.
          </div>
        )}

        <div className="absolute bottom-4 right-4 fd-pill">
          Showing: {mode.toUpperCase()}
        </div>
      </div>
    </div>
  );
};
