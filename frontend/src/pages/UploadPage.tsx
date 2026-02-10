import React, { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { UploadCloud, AlertCircle } from "lucide-react";
import { detectImage, detectVideo, detectVideoAsync, getVideoJob } from "../services/api";
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
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [previewFailed, setPreviewFailed] = useState(false);
  const [persistedFilename, setPersistedFilename] = useState<string | null>(null);
  const [persistedMediaType, setPersistedMediaType] = useState<
    "video" | "image" | null
  >(null);
  const persistJob = (data: Record<string, unknown> | null) => {
    try {
      if (!data) {
        localStorage.removeItem("fd_video_job");
      } else {
        localStorage.setItem("fd_video_job", JSON.stringify(data));
      }
    } catch {
      // ignore storage errors
    }
  };
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollerRef = useRef<number | null>(null);
  const cancelledRef = useRef(false);
  const navigate = useNavigate();

  useEffect(() => {
    return () => {
      cancelledRef.current = true;
      if (pollerRef.current) {
        window.clearTimeout(pollerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!file && activeAnalysisFile) {
      setFile(activeAnalysisFile);
      const refreshedPreview = URL.createObjectURL(activeAnalysisFile);
      setPreview(refreshedPreview);
      setPreviewFailed(false);
      activeAnalysisPreview = refreshedPreview;
      setPersistedFilename(activeAnalysisFile.name);
    }
  }, [file]);

  useEffect(() => {
    const readJob = () => {
      try {
        const raw = localStorage.getItem("fd_video_job");
        if (!raw) return null;
        return JSON.parse(raw) as {
          jobId: string;
          status: string;
          filename?: string;
          resultId?: number;
          mediaType?: "video" | "image";
          previewUrl?: string | null;
        };
      } catch {
        return null;
      }
    };

    const writeJob = (value: Record<string, unknown> | null) => {
      try {
        if (!value) {
          localStorage.removeItem("fd_video_job");
        } else {
          localStorage.setItem("fd_video_job", JSON.stringify(value));
        }
      } catch {
        // ignore storage errors
      }
    };

    const readSeenJobId = () => {
      try {
        return localStorage.getItem("fd_video_job_seen");
      } catch {
        return null;
      }
    };

    const writeSeenJobId = (jobId: string) => {
      try {
        localStorage.setItem("fd_video_job_seen", jobId);
      } catch {
        // ignore
      }
    };

    const job = readJob();
    if (!job?.jobId) return;

    if (job.status === "completed" && job.resultId) {
      const seenJobId = readSeenJobId();
      if (seenJobId !== job.jobId) {
        writeSeenJobId(job.jobId);
        writeJob(null);
        navigate(`/videos/${job.resultId}`);
      }
      return;
    }

    if (job.status === "failed") {
      setError("Video analysis failed.");
      writeJob(null);
      return;
    }

    setLoading(true);
    setJobStatus(job.status);
    setPersistedMediaType(job.mediaType === "image" ? "image" : "video");
    if (job.previewUrl) {
      setPreview(job.previewUrl);
      setPreviewFailed(false);
    }
    if (job.filename) {
      setPersistedFilename(job.filename);
    }

    const poll = async () => {
      if (cancelledRef.current) return;
      try {
        const status = await getVideoJob(job.jobId);
        setJobStatus(status.status);
        const updated = {
          jobId: job.jobId,
          status: status.status,
          filename: status.filename || job.filename,
          resultId: status.result?.id ?? job.resultId,
          mediaType: job.mediaType,
          previewUrl: job.previewUrl,
        };
        writeJob(updated);

        if (status.status === "completed" && status.result?.id) {
          setLoading(false);
          setJobStatus(null);
          writeJob(null);
          writeSeenJobId(job.jobId);
          navigate(`/videos/${status.result.id}`);
          return;
        }
        if (status.status === "failed") {
          setError(status.error || "Video analysis failed.");
          setLoading(false);
          setJobStatus(null);
          writeJob(null);
          return;
        }

        pollerRef.current = window.setTimeout(poll, 2000);
      } catch {
      setError("Failed to fetch job status.");
      setLoading(false);
      setJobStatus(null);
    }
  };

    poll();
  }, [navigate]);

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
        setJobStatus(null);
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
      setPreviewFailed(false);
      setPersistedFilename(selectedFile.name);
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
      setPreviewFailed(false);
      setPersistedFilename(selectedFile.name);
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyse = async () => {
    if (!file || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setJobStatus(null);
    activeAnalysisFile = file;
    activeAnalysisPreview = preview;
    const isVideo = (file.type || "").startsWith("video/");
    activeAnalysisType = isVideo ? "video" : "image";

    try {
      if (isVideo) {
        const { job_id } = await detectVideoAsync(file);
        setJobStatus("queued");
        cancelledRef.current = false;
        const filename = file?.name || "video";
        const previewUrl = preview || null;
        setPersistedMediaType("video");
        setPersistedFilename(filename);
        persistJob({
          jobId: job_id,
          status: "queued",
          filename,
          updatedAt: new Date().toISOString(),
          mediaType: "video",
          previewUrl,
        });

        const poll = async () => {
          if (cancelledRef.current) return;
          try {
            const job = await getVideoJob(job_id);
            setJobStatus(job.status);
            persistJob({
              jobId: job_id,
              status: job.status,
              filename: job.filename || filename,
              resultId: job.result?.id,
              updatedAt: new Date().toISOString(),
              mediaType: "video",
              previewUrl,
            });
            if (job.status === "completed" && job.result?.id) {
              setLoading(false);
              setJobStatus(null);
              activeAnalysisPromise = null;
              activeAnalysisType = null;
              activeAnalysisFile = null;
              activeAnalysisPreview = null;
              persistJob(null);
              navigate(`/videos/${job.result.id}`);
              return;
            }
            if (job.status === "failed") {
              setError(job.error || "Video analysis failed.");
              setLoading(false);
              setJobStatus(null);
              activeAnalysisPromise = null;
              activeAnalysisType = null;
              activeAnalysisFile = null;
              activeAnalysisPreview = null;
              persistJob(null);
              return;
            }
            pollerRef.current = window.setTimeout(poll, 2000);
          } catch {
            setError("Failed to fetch job status.");
            setLoading(false);
            setJobStatus(null);
          }
        };
        await poll();
      } else {
        activeAnalysisPromise = detectImage(file);
        const data = await activeAnalysisPromise;
        const imageResult = data as AnalysisResult;
        if (imageResult.id) {
          persistJob(null);
          navigate(`/records/${imageResult.id}`);
          return;
        }
        setResult(imageResult);
        setPersistedMediaType("image");
        setPersistedFilename(imageResult.filename || null);
      }
    } catch {
      setError("Failed to analyse media. Please ensure backend is running.");
    } finally {
      if (!isVideo) {
        setLoading(false);
        activeAnalysisPromise = null;
        activeAnalysisType = null;
        activeAnalysisFile = null;
        activeAnalysisPreview = null;
        setPersistedMediaType(null);
      }
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setLoading(false);
    setJobStatus(null);
    setPersistedMediaType(null);
    setPreviewFailed(false);
    setPersistedFilename(null);
    cancelledRef.current = true;
    if (pollerRef.current) {
      window.clearTimeout(pollerRef.current);
      pollerRef.current = null;
    }
    persistJob(null);
    activeAnalysisPromise = null;
    activeAnalysisType = null;
    activeAnalysisFile = null;
    activeAnalysisPreview = null;
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      {/* Header */}
      <div>
        <div className="fd-kicker mb-2">Evidence Intake</div>
        <h1 className="text-3xl font-semibold text-slate-100 mb-2 fd-title">
          New Analysis
        </h1>
        <p className="text-slate-400 max-w-2xl">
          Upload an image or video to detect AI-generated content, manipulate
          artifacts, and metadata anomalies.
        </p>
      </div>

      {/* Upload Zone - Only show if no result yet */}
      {!result && (
        <div className="animate-fade-in">
          <div
            className={`fd-card border-dashed border-2 p-10 flex flex-col items-center justify-center transition-all duration-200 
              ${
                preview
                  ? "border-cyan-400/40 bg-cyan-900/10"
                  : "border-slate-800/80 bg-slate-950/30 hover:border-slate-600 hover:bg-slate-900/40"
              }`}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
          >
            {preview && !previewFailed ? (
              <div className="relative group">
                {(file?.type || "").startsWith("video/") ? (
                  <video
                    src={preview}
                    controls
                    onError={() => setPreviewFailed(true)}
                    className="max-h-[300px] rounded-lg shadow-xl"
                  />
                ) : (
                  <img
                    src={preview}
                    alt="Preview"
                    onError={() => setPreviewFailed(true)}
                    className="max-h-[300px] rounded-lg shadow-xl"
                  />
                )}
                <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded-lg">
                  <Button variant="secondary" onClick={reset}>
                    Change File
                  </Button>
                </div>
              </div>
            ) : persistedFilename ? (
              <div className="text-center">
                <div className="w-20 h-20 bg-slate-900 rounded-full flex items-center justify-center mx-auto mb-4 text-cyan-300">
                  <UploadCloud size={40} />
                </div>
                <h3 className="text-xl font-semibold text-slate-200 mb-2">
                  {persistedFilename}
                </h3>
                <p className="text-slate-500 mb-6 max-w-sm mx-auto">
                  Preview not available. Analysis is still running.
                </p>
                <div className="flex justify-center">
                  <Button variant="secondary" onClick={reset}>
                    Change File
                  </Button>
                </div>
              </div>
            ) : (
              <div className="text-center">
                <div className="w-20 h-20 bg-slate-900 rounded-full flex items-center justify-center mx-auto mb-4 text-cyan-300">
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
                Analysing{" "}
                {file?.type?.startsWith("video/")
                  ? "Video"
                  : file
                  ? "Image"
                  : persistedMediaType === "video"
                  ? "Video"
                  : "Image"}
                ...
              </h3>
              <p className="text-slate-500 mt-2">
                Running CNN detection, ELA, and metadata extraction.
              </p>
              {jobStatus && (
                <p className="text-slate-400 mt-2 text-sm">
                  Status: {jobStatus}
                </p>
              )}
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
