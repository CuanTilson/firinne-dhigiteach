import React, { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { UploadCloud, AlertCircle } from "lucide-react";
import { detectImage, detectVideo } from "../services/api";
import type { AnalysisResult, VideoAnalysisDetail } from "../types";
import { AnalysisDashboard } from "../components/AnalysisDashboard";
import { Button } from "../components/ui/Button";

let activeAnalysisPromise: Promise<AnalysisResult | VideoAnalysisDetail> | null =
  null;
let activeAnalysisType: "image" | "video" | null = null;
let activeAnalysisFile: File | null = null;
let activeAnalysisPreview: string | null = null;

export const UploadPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!activeAnalysisPromise) return;

    setFile(activeAnalysisFile);
    setPreview(activeAnalysisPreview);
    setLoading(true);
    setError(null);

    activeAnalysisPromise
      .then((data) => {
        if (activeAnalysisType === "video") {
          const video = data as VideoAnalysisDetail;
          navigate(`/videos/${video.id}`);
          return;
        }
        setResult(data as AnalysisResult);
      })
      .catch(() => {
        setError("Failed to analyse media. Please ensure backend is running.");
      })
      .finally(() => {
        setLoading(false);
        activeAnalysisPromise = null;
        activeAnalysisType = null;
        activeAnalysisFile = null;
        activeAnalysisPreview = null;
      });
  }, [navigate]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selectedFile = e.dataTransfer.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyse = async () => {
    if (!file || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    activeAnalysisFile = file;
    activeAnalysisPreview = preview;
    const isVideo = (file.type || "").startsWith("video/");
    activeAnalysisType = isVideo ? "video" : "image";

    try {
      activeAnalysisPromise = isVideo ? detectVideo(file) : detectImage(file);
      const data = await activeAnalysisPromise;
      if (isVideo) {
        const video = data as VideoAnalysisDetail;
        navigate(`/videos/${video.id}`);
      } else {
        setResult(data as AnalysisResult);
      }
    } catch {
      setError("Failed to analyse media. Please ensure backend is running.");
    } finally {
      setLoading(false);
      activeAnalysisPromise = null;
      activeAnalysisType = null;
      activeAnalysisFile = null;
      activeAnalysisPreview = null;
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setLoading(false);
    activeAnalysisPromise = null;
    activeAnalysisType = null;
    activeAnalysisFile = null;
    activeAnalysisPreview = null;
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">New Analysis</h1>
        <p className="text-slate-400">
          Upload an image or video to detect AI-generated content, manipulate
          artifacts, and metadata anomalies.
        </p>
      </div>

      {/* Upload Zone - Only show if no result yet */}
      {!result && (
        <div className="mb-8 animate-fade-in">
          <div
            className={`border-2 border-dashed rounded-2xl p-10 flex flex-col items-center justify-center transition-all duration-200 
              ${
                preview
                  ? "border-cyan-500/50 bg-cyan-900/5"
                  : "border-slate-700 bg-slate-800/50 hover:border-slate-500 hover:bg-slate-800"
              }`}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
          >
            {preview ? (
              <div className="relative group">
                {(file?.type || "").startsWith("video/") ? (
                  <video
                    src={preview}
                    controls
                    className="max-h-[300px] rounded-lg shadow-xl"
                  />
                ) : (
                  <img
                    src={preview}
                    alt="Preview"
                    className="max-h-[300px] rounded-lg shadow-xl"
                  />
                )}
                <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded-lg">
                  <Button variant="secondary" onClick={reset}>
                    Change File
                  </Button>
                </div>
              </div>
            ) : (
              <div className="text-center">
                <div className="w-20 h-20 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4 text-cyan-500">
                  <UploadCloud size={40} />
                </div>
                <h3 className="text-xl font-semibold text-slate-200 mb-2">
                  Drag & Drop Media
                </h3>
                <p className="text-slate-500 mb-6 max-w-sm mx-auto">
                  Supported formats: JPEG, PNG, WEBP, TIFF, MP4, MOV. Max file
                  size: 200MB, max length: 3 minutes.
                </p>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  className="hidden"
                  accept="image/*,video/*"
                />
                <div className="flex justify-center">
                  <Button onClick={() => fileInputRef.current?.click()}>
                    Browse Files
                  </Button>
                </div>
              </div>
            )}
          </div>

          {file && !loading && (
            <div className="mt-6 flex justify-center">
              <Button
                onClick={handleAnalyse}
                className="px-8 py-3 text-lg shadow-cyan-500/20 shadow-lg"
              >
                Run Analysis
              </Button>
            </div>
          )}

          {loading && (
            <div className="mt-8 text-center">
              <div className="inline-block animate-spin rounded-full h-10 w-10 border-4 border-cyan-500 border-r-transparent mb-4"></div>
              <h3 className="text-xl font-medium text-slate-200">
                Analysing {file?.type?.startsWith("video/") ? "Video" : "Image"}...
              </h3>
              <p className="text-slate-500 mt-2">
                Running CNN detection, ELA, and metadata extraction.
              </p>
            </div>
          )}

          {error && (
            <div className="mt-6 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-3 text-red-200 max-w-2xl mx-auto">
              <AlertCircle />
              {error}
            </div>
          )}
        </div>
      )}

      {/* Results View */}
      {result && (
        <div>
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-slate-200">
              Analysis Results
            </h2>
            <Button variant="secondary" onClick={reset}>
              Analyse Another
            </Button>
          </div>
          <AnalysisDashboard result={result} />
        </div>
      )}
    </div>
  );
};
