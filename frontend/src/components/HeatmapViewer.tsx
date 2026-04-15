import React, { useMemo, useState } from "react";
import { Eye, Layers, Zap, Activity, Aperture } from "lucide-react";

interface HeatmapViewerProps {
  originalUrl?: string;
  elaUrl?: string;
  gradCamUrl?: string;
  noiseUrl?: string;
  jpegQualityUrl?: string;
}

type ViewMode = "original" | "ela" | "gradcam" | "noise" | "jpeg";

interface ViewOption {
  key: ViewMode;
  label: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
  src?: string;
}

export const HeatmapViewer: React.FC<HeatmapViewerProps> = ({
  originalUrl,
  elaUrl,
  gradCamUrl,
  noiseUrl,
  jpegQualityUrl,
}) => {
  const views = useMemo<ViewOption[]>(
    () => [
      { key: "original", icon: Eye, label: "Original", src: originalUrl },
      { key: "ela", icon: Layers, label: "ELA", src: elaUrl },
      { key: "gradcam", icon: Zap, label: "GradCAM", src: gradCamUrl },
      { key: "noise", icon: Activity, label: "Noise", src: noiseUrl },
      { key: "jpeg", icon: Aperture, label: "JPEG", src: jpegQualityUrl },
    ],
    [originalUrl, elaUrl, gradCamUrl, noiseUrl, jpegQualityUrl],
  );

  const firstAvailableMode = views.find((view) => view.src)?.key ?? "original";

  const [mode, setMode] = useState<ViewMode>(firstAvailableMode);

  const activeView =
    views.find((view) => view.key === mode) ??
    views.find((view) => view.key === "original") ??
    views[0];

  const src = activeView?.src;

  return (
    <div className="overflow-hidden rounded-3xl border border-slate-800/80 bg-slate-950/55 shadow-[0_10px_30px_rgba(2,6,23,0.28)]">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 px-4 py-4">
        <div>
          <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500">
            Visual Evidence
          </div>
          <h3 className="mt-1 text-lg font-semibold text-slate-100">
            Visual Analysis
          </h3>
        </div>

        <div className="flex flex-wrap gap-2">
          {views.map(({ key, icon: Icon, label, src: optionSrc }) => {
            const active = mode === key;
            const unavailable = !optionSrc;

            return (
              <button
                key={key}
                type="button"
                onClick={() => setMode(key)}
                aria-pressed={active}
                aria-label={`Show ${label} view`}
                className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium transition ${
                  active
                    ? "border-cyan-400/40 bg-cyan-500/20 text-cyan-200"
                    : "border-slate-800 text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                } ${
                  unavailable ? "opacity-50" : ""
                } focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-slate-900`}
              >
                <Icon size={14} />
                {label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="relative flex min-h-[420px] items-center justify-center bg-slate-950/70">
        <div
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: "radial-gradient(#334155 1px, transparent 1px)",
            backgroundSize: "24px 24px",
          }}
        />

        {src ? (
          <img
            src={src}
            alt={`${activeView.label} analysis view`}
            className="max-h-[560px] max-w-full object-contain p-6"
          />
        ) : (
          <div className="rounded-2xl border border-slate-800 bg-slate-950/80 px-6 py-5 text-center text-sm text-slate-500">
            No image available for the {activeView.label} view.
          </div>
        )}

        <div className="absolute bottom-4 right-4 rounded-full border border-slate-800 bg-slate-950/85 px-3 py-1 text-xs text-slate-300">
          Showing: {activeView.label}
        </div>
      </div>
    </div>
  );
};
