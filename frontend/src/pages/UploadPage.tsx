import React, { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { UploadCloud, AlertCircle } from "lucide-react";
import {
  detectAudio,
  detectImage,
  detectVideoAsync,
  getVideoJob,
} from "../services/api";
import type {
  AnalysisResult,
  AudioAnalysisDetail,
  VideoAnalysisDetail,
} from "../types";
import { AnalysisDashboard } from "../components/AnalysisDashboard";
import { Button } from "../components/ui/Button";

type PersistedJob = {
  jobId: string;
  status: string;
  filename?: string;
  resultId?: number;
  mediaType?: "video" | "image" | "audio";
};

const JOB_STORAGE_KEY = "fd_video_job";
const JOB_SEEN_KEY = "fd_video_job_seen";

const readPersistedJob = (): PersistedJob | null => {
  try {
    const raw = localStorage.getItem(JOB_STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as PersistedJob;
  } catch {
    return null;
  }
};

const writePersistedJob = (job: PersistedJob | null) => {
  try {
    if (!job) {
      localStorage.removeItem(JOB_STORAGE_KEY);
    } else {
      localStorage.setItem(JOB_STORAGE_KEY, JSON.stringify(job));
    }
  } catch {
    // ignore storage errors
  }
};

const readSeenJobId = () => {
  try {
    return localStorage.getItem(JOB_SEEN_KEY);
  } catch {
    return null;
  }
};

const writeSeenJobId = (jobId: string) => {
  try {
    localStorage.setItem(JOB_SEEN_KEY, jobId);
  } catch {
    // ignore storage errors
  }
};

const getMediaTypeFromFile = (file: File): "video" | "audio" | "image" => {
  if ((file.type || "").startsWith("video/")) return "video";
  if ((file.type || "").startsWith("audio/")) return "audio";
  return "image";
};

const formatFileSize = (bytes: number) => {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const getLoadingLabel = (
  mediaType: "video" | "audio" | "image" | null,
  jobStatus: string | null,
) => {
  if (mediaType === "video") {
    if (jobStatus === "queued") return "Video queued for analysis...";
    if (jobStatus === "processing" || jobStatus === "running") {
      return "Analysing video...";
    }
    return "Uploading video...";
  }

  if (mediaType === "audio") return "Analysing audio...";
  return "Analysing image...";
};

export const UploadPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [previewFailed, setPreviewFailed] = useState(false);
  const [persistedFilename, setPersistedFilename] = useState<string | null>(
    null,
  );
  const [persistedMediaType, setPersistedMediaType] = useState<
    "video" | "image" | "audio" | null
  >(null);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollerRef = useRef<number | null>(null);
  const previewUrlRef = useRef<string | null>(null);
  const cancelledRef = useRef(false);

  const navigate = useNavigate();

  const effectiveMediaType = useMemo(() => {
    if (file) return getMediaTypeFromFile(file);
    return persistedMediaType;
  }, [file, persistedMediaType]);

  const clearPoller = () => {
    if (pollerRef.current) {
      window.clearTimeout(pollerRef.current);
      pollerRef.current = null;
    }
  };

  const replacePreview = (nextPreview: string | null) => {
    if (previewUrlRef.current && previewUrlRef.current.startsWith("blob:")) {
      URL.revokeObjectURL(previewUrlRef.current);
    }
    previewUrlRef.current = nextPreview;
    setPreview(nextPreview);
  };

  const clearPersistedJobState = () => {
    writePersistedJob(null);
    setJobStatus(null);
    setPersistedFilename(null);
    setPersistedMediaType(null);
  };

  const selectFile = (selectedFile: File) => {
    clearPoller();
    cancelledRef.current = false;

    const objectUrl = URL.createObjectURL(selectedFile);

    setFile(selectedFile);
    replacePreview(objectUrl);
    setPreviewFailed(false);
    setPersistedFilename(selectedFile.name);
    setPersistedMediaType(getMediaTypeFromFile(selectedFile));
    setResult(null);
    setError(null);
    setJobStatus(null);
  };

  const reset = () => {
    cancelledRef.current = true;
    clearPoller();

    setFile(null);
    replacePreview(null);
    setResult(null);
    setError(null);
    setLoading(false);
    setJobStatus(null);
    setPersistedMediaType(null);
    setPreviewFailed(false);
    setPersistedFilename(null);
    setIsDragging(false);

    clearPersistedJobState();

    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  useEffect(() => {
    return () => {
      cancelledRef.current = true;
      clearPoller();
      if (previewUrlRef.current && previewUrlRef.current.startsWith("blob:")) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const job = readPersistedJob();
    if (!job?.jobId) return;

    if (job.status === "completed" && job.resultId) {
      const seenJobId = readSeenJobId();
      if (seenJobId !== job.jobId) {
        writeSeenJobId(job.jobId);
        writePersistedJob(null);
        navigate(`/videos/${job.resultId}`);
      }
      return;
    }

    if (job.status === "failed") {
      setError("Video analysis failed.");
      writePersistedJob(null);
      return;
    }

    cancelledRef.current = false;
    setLoading(true);
    setJobStatus(job.status);
    setPersistedFilename(job.filename || null);
    setPersistedMediaType(job.mediaType || "video");
    setPreviewFailed(true);

    const poll = async () => {
      if (cancelledRef.current) return;

      try {
        const status = await getVideoJob(job.jobId);
        if (cancelledRef.current) return;

        setJobStatus(status.status);

        writePersistedJob({
          jobId: job.jobId,
          status: status.status,
          filename: status.filename || job.filename,
          resultId: status.result?.id ?? job.resultId,
          mediaType: job.mediaType || "video",
        });

        if (status.status === "completed" && status.result?.id) {
          setLoading(false);
          clearPersistedJobState();
          writeSeenJobId(job.jobId);
          navigate(`/videos/${status.result.id}`);
          return;
        }

        if (status.status === "failed") {
          setError(status.error || "Video analysis failed.");
          setLoading(false);
          clearPersistedJobState();
          return;
        }

        pollerRef.current = window.setTimeout(poll, 2000);
      } catch {
        setError("Failed to fetch video job status.");
        setLoading(false);
        setJobStatus(null);
      }
    };

    poll();

    return () => {
      cancelledRef.current = true;
      clearPoller();
    };
  }, [navigate]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files?.[0]) {
      selectFile(event.target.files[0]);
    }
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setIsDragging(false);

    if (event.dataTransfer.files?.[0]) {
      selectFile(event.dataTransfer.files[0]);
    }
  };

  const handleAnalyse = async () => {
    if (!file || loading) return;

    clearPoller();
    cancelledRef.current = false;

    const mediaType = getMediaTypeFromFile(file);

    setLoading(true);
    setError(null);
    setResult(null);
    setJobStatus(null);
    setPersistedMediaType(mediaType);
    setPersistedFilename(file.name);

    try {
      if (mediaType === "video") {
        const { job_id } = await detectVideoAsync(file);

        setJobStatus("queued");

        writePersistedJob({
          jobId: job_id,
          status: "queued",
          filename: file.name,
          mediaType: "video",
        });

        const poll = async () => {
          if (cancelledRef.current) return;

          try {
            const job = await getVideoJob(job_id);
            if (cancelledRef.current) return;

            setJobStatus(job.status);

            writePersistedJob({
              jobId: job_id,
              status: job.status,
              filename: job.filename || file.name,
              resultId: job.result?.id,
              mediaType: "video",
            });

            if (job.status === "completed" && job.result?.id) {
              setLoading(false);
              clearPersistedJobState();
              navigate(`/videos/${job.result.id}`);
              return;
            }

            if (job.status === "failed") {
              setError(job.error || "Video analysis failed.");
              setLoading(false);
              clearPersistedJobState();
              return;
            }

            pollerRef.current = window.setTimeout(poll, 2000);
          } catch {
            setError("Failed to fetch video job status.");
            setLoading(false);
            setJobStatus(null);
          }
        };

        poll();
        return;
      }

      if (mediaType === "audio") {
        const audioResult = await detectAudio(file);
        navigate(`/audio/${audioResult.id}`);
        return;
      }

      const imageResult = await detectImage(file);
      if (imageResult.id) {
        navigate(`/records/${imageResult.id}`);
        return;
      }

      setResult(imageResult);
      setPersistedMediaType("image");
      setPersistedFilename(imageResult.filename || file.name);
    } catch {
      setError("Failed to analyse media. Please try again.");
    } finally {
      if (mediaType !== "video") {
        setLoading(false);
        setJobStatus(null);
      }
    }
  };

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 md:px-6">
      <div className="space-y-6">
        <section className="relative overflow-hidden rounded-3xl border border-slate-800/80 bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.10),_transparent_35%),linear-gradient(180deg,rgba(2,6,23,0.96),rgba(2,6,23,0.82))] p-6 shadow-[0_0_0_1px_rgba(15,23,42,0.5)]">
          <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,rgba(148,163,184,0.04)_1px,transparent_1px),linear-gradient(to_bottom,rgba(148,163,184,0.04)_1px,transparent_1px)] bg-[size:32px_32px] opacity-30" />

          <div className="relative">
            <div className="mb-2 text-[11px] uppercase tracking-[0.28em] text-cyan-300/80">
              Evidence Intake
            </div>
            <h1 className="text-3xl font-semibold tracking-tight text-slate-50">
              New Analysis
            </h1>
            <p className="mt-2 max-w-3xl text-slate-400">
              Upload an image, video, or audio file to detect AI-generated
              content, manipulation artefacts, and metadata anomalies.
            </p>
          </div>
        </section>

        {!result && (
          <div className="animate-fade-in space-y-6">
            <section
              className={`rounded-3xl border-2 border-dashed p-10 transition ${
                isDragging
                  ? "border-cyan-400/60 bg-cyan-900/15"
                  : preview || persistedFilename
                    ? "border-cyan-400/30 bg-cyan-900/10"
                    : "border-slate-800/80 bg-slate-950/30 hover:border-slate-600 hover:bg-slate-900/40"
              }`}
              role="button"
              tabIndex={0}
              aria-label="Upload media evidence"
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onKeyDown={(event) => {
                if ((event.key === "Enter" || event.key === " ") && !preview) {
                  event.preventDefault();
                  fileInputRef.current?.click();
                }
              }}
            >
              {preview && !previewFailed ? (
                <div className="space-y-6 text-center">
                  <div className="relative inline-block">
                    {(file?.type || "").startsWith("video/") ? (
                      <video
                        src={preview}
                        controls
                        onError={() => setPreviewFailed(true)}
                        className="max-h-[320px] rounded-xl shadow-xl"
                      />
                    ) : (file?.type || "").startsWith("audio/") ? (
                      <div className="min-w-[320px] rounded-xl border border-slate-800 bg-slate-950 p-6 shadow-xl">
                        <div className="mb-3 text-sm text-slate-400">
                          {file?.name}
                        </div>
                        <audio
                          src={preview}
                          controls
                          onError={() => setPreviewFailed(true)}
                          className="w-full"
                        />
                      </div>
                    ) : (
                      <img
                        src={preview}
                        alt="Preview"
                        onError={() => setPreviewFailed(true)}
                        className="max-h-[320px] rounded-xl shadow-xl"
                      />
                    )}
                  </div>

                  {file ? (
                    <div className="space-y-2">
                      <div className="text-lg font-semibold text-slate-200">
                        {file.name}
                      </div>
                      <div className="text-sm text-slate-500">
                        {effectiveMediaType} · {formatFileSize(file.size)}
                      </div>
                    </div>
                  ) : null}

                  <div className="flex justify-center gap-3">
                    <Button onClick={handleAnalyse} disabled={loading}>
                      Run Analysis
                    </Button>
                    <Button
                      variant="secondary"
                      onClick={reset}
                      disabled={loading}
                    >
                      Change File
                    </Button>
                  </div>
                </div>
              ) : persistedFilename ? (
                <div className="text-center">
                  <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-full bg-slate-900 text-cyan-300">
                    <UploadCloud size={40} />
                  </div>
                  <h3 className="mb-2 text-xl font-semibold text-slate-200">
                    {persistedFilename}
                  </h3>
                  <p className="mx-auto mb-6 max-w-sm text-slate-500">
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
                  <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-full bg-slate-900 text-cyan-300">
                    <UploadCloud size={40} />
                  </div>
                  <h3 className="mb-2 text-xl font-semibold text-slate-200">
                    Drag &amp; Drop Media
                  </h3>
                  <p className="mx-auto mb-6 max-w-md text-slate-500">
                    Supported formats: JPEG, PNG, WEBP, TIFF, MP4, MOV, WAV,
                    MP3, M4A, FLAC. Max file size: 200MB, max video length: 3
                    minutes.
                  </p>

                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    className="hidden"
                    accept="image/*,video/*,audio/*,.wav,.mp3,.m4a,.flac"
                    aria-label="Browse media files"
                  />

                  <div className="flex justify-center">
                    <Button onClick={() => fileInputRef.current?.click()}>
                      Browse Files
                    </Button>
                  </div>
                </div>
              )}
            </section>

            {loading ? (
              <div className="text-center" aria-live="polite">
                <div className="mb-4 inline-block h-10 w-10 animate-spin rounded-full border-4 border-cyan-500 border-r-transparent" />
                <h3 className="text-xl font-medium text-slate-200">
                  {getLoadingLabel(effectiveMediaType, jobStatus)}
                </h3>
                <p className="mt-2 text-slate-500">
                  Running forensic checks and metadata extraction.
                </p>
                {jobStatus ? (
                  <p className="mt-2 text-sm text-slate-400">
                    Status: {jobStatus}
                  </p>
                ) : null}
              </div>
            ) : null}

            {error ? (
              <div
                className="mx-auto flex max-w-2xl items-center gap-3 rounded-xl border border-red-500/20 bg-red-500/10 p-4 text-red-200"
                aria-live="assertive"
              >
                <AlertCircle size={18} />
                {error}
              </div>
            ) : null}
          </div>
        )}

        {result ? (
          <div>
            <div className="mb-6 flex items-center justify-between">
              <h2 className="text-2xl font-bold text-slate-200">
                Analysis Results
              </h2>
              <Button variant="secondary" onClick={reset}>
                Analyse Another
              </Button>
            </div>

            <AnalysisDashboard result={result} />
          </div>
        ) : null}
      </div>
    </div>
  );
};
