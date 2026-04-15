import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getVideoById } from "../services/api";
import type { VideoAnalysisDetail } from "../types";
import { fixPath } from "../constants";
import { ArrowLeft } from "lucide-react";

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : null;

const formatClassification = (value?: string | null) => {
  if (!value) return "Unavailable";
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
};

const formatUnknown = (value: unknown, suffix?: string) => {
  if (typeof value === "number") {
    const formatted = Number.isInteger(value)
      ? String(value)
      : value.toFixed(3);
    return suffix ? `${formatted} ${suffix}` : formatted;
  }
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string" && value.trim()) {
    return suffix ? `${value} ${suffix}` : value;
  }
  return "Unavailable";
};

const formatNestedStat = (
  source: Record<string, unknown>,
  key: string,
  stat: string,
) => {
  const nested = asRecord(source[key]);
  return formatUnknown(nested?.[stat]);
};

const Section = ({
  title,
  children,
  avoidBreak = true,
}: {
  title: string;
  children: React.ReactNode;
  avoidBreak?: boolean;
}) => (
  <section
    className="rounded-lg border border-slate-200 p-4"
    style={avoidBreak ? { breakInside: "avoid" } : undefined}
  >
    <div className="mb-3 text-xs uppercase tracking-[0.18em] text-slate-500">
      {title}
    </div>
    {children}
  </section>
);

const KV = ({
  label,
  value,
  mono = false,
}: {
  label: string;
  value?: string | null;
  mono?: boolean;
}) => (
  <div>
    <div className="text-xs uppercase text-slate-500">{label}</div>
    <div className={`${mono ? "font-mono text-xs" : "font-medium"} break-all`}>
      {value || "Unavailable"}
    </div>
  </div>
);

const PrintDecisionSummary = ({
  verdict,
  scoreValue,
  rationale,
  note,
}: {
  verdict: string;
  scoreValue: string;
  rationale: string[];
  note?: string;
}) => (
  <Section title="Decision Summary">
    <div className="grid grid-cols-1 gap-4 md:grid-cols-[1fr_180px]">
      <div>
        <div className="text-xs uppercase tracking-[0.16em] text-slate-500">
          Final Verdict
        </div>
        <div className="mt-1 text-2xl font-semibold text-slate-900">
          {verdict}
        </div>
      </div>
      <div className="rounded-lg border border-slate-200 p-3">
        <div className="text-xs uppercase tracking-[0.16em] text-slate-500">
          Forensic Score
        </div>
        <div className="mt-1 text-2xl font-semibold text-slate-900">
          {scoreValue}
        </div>
      </div>
    </div>

    <div className="mt-4">
      <div className="mb-2 text-xs uppercase tracking-[0.16em] text-slate-500">
        Basis for Current Decision
      </div>
      <ul className="list-disc space-y-1 pl-5 text-sm text-slate-800">
        {rationale.map((item, index) => (
          <li key={`${item}-${index}`}>{item}</li>
        ))}
      </ul>
    </div>

    {note ? <div className="mt-4 text-sm text-slate-600">{note}</div> : null}
  </Section>
);

const PrintAppliedSettings = ({
  settings,
}: {
  settings?: Record<string, unknown> | null;
}) => {
  const pipeline = asRecord(settings?.pipeline);
  const thresholds = asRecord(settings?.thresholds);
  const imageBands = asRecord(thresholds?.classification_bands);
  const audioBands = asRecord(thresholds?.audio_classification_bands);
  const paths = asRecord(settings?.paths);

  return (
    <Section title="Applied Settings">
      {!settings ? (
        <div className="text-sm text-slate-600">
          No settings snapshot was stored for this record.
        </div>
      ) : (
        <div className="space-y-4">
          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
              Pipeline
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <KV
                label="Pipeline Version"
                value={formatUnknown(pipeline?.pipeline_version)}
              />
              <KV
                label="Image Detector"
                value={formatUnknown(pipeline?.image_detector)}
              />
              <KV
                label="Model Version"
                value={formatUnknown(pipeline?.model_version)}
              />
              <KV
                label="Dataset Version"
                value={formatUnknown(pipeline?.dataset_version)}
              />
            </div>
          </div>

          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
              Thresholds
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <KV
                label="Image AI Threshold"
                value={formatUnknown(imageBands?.ai_likely_min)}
              />
              <KV
                label="Image Real Threshold"
                value={formatUnknown(imageBands?.real_likely_max)}
              />
              <KV
                label="Audio AI Threshold"
                value={formatUnknown(audioBands?.ai_likely_min)}
              />
              <KV
                label="Audio Real Threshold"
                value={formatUnknown(audioBands?.real_likely_max)}
              />
              <KV
                label="Video Sample Frames"
                value={formatUnknown(thresholds?.video_sample_frames)}
              />
            </div>
          </div>

          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
              Runtime
            </div>
            <KV
              label="FFmpeg Path"
              value={formatUnknown(paths?.ffmpeg_path)}
              mono
            />
          </div>
        </div>
      )}
    </Section>
  );
};

export const PrintVideoPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<VideoAnalysisDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDetail = async () => {
      if (!id) {
        setError("No record id was provided.");
        setLoading(false);
        return;
      }

      try {
        const data = await getVideoById(Number(id));
        setResult(data);
      } catch {
        setError("Could not load video analysis details.");
      } finally {
        setLoading(false);
      }
    };

    fetchDetail();
  }, [id]);

  const handleGeneratePdf = () => window.print();

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center text-slate-500">
        Loading report...
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 text-red-500">
        <p>{error || "Record not found"}</p>
        <Link to="/history" className="text-cyan-600 hover:underline">
          Back to History
        </Link>
      </div>
    );
  }

  const frames = result.frames?.slice(0, 8) || [];
  const representativeFrame =
    result.frames?.find(
      (frame) =>
        frame.saved_path ||
        frame.gradcam_heatmap ||
        frame.ela_heatmap ||
        frame.noise_residual?.noise_heatmap_path ||
        frame.jpeg_qtables?.double_compression?.jpeg_quality_heatmap_path,
    ) ??
    result.frames?.[0] ??
    null;
  const videoMeta = asRecord(result.video_metadata);
  const hashes = asRecord(videoMeta?.hashes);

  const audio = result.audio_analysis ?? null;
  const audioMeta = asRecord(audio?.audio_metadata);
  const audioFeatures = asRecord(audio?.audio_features);
  const audioSegments = asRecord(audioFeatures?.segment_summary);
  const audioHashes = audio?.file_integrity ?? null;

  const integrityNote =
    hashes?.sha256 && hashes?.md5
      ? "Integrity hashes were recorded for the source video."
      : "Integrity hashes were not fully available.";

  return (
    <div className="min-h-screen bg-slate-100 px-4 py-8 text-slate-900 print:bg-white print:p-0">
      <div className="mx-auto max-w-5xl space-y-6 rounded-xl border border-slate-200 bg-white p-6 print:border-0 print:p-0">
        <div
          className="flex items-start justify-between border-b border-slate-200 pb-4"
          style={{ breakInside: "avoid" }}
        >
          <div>
            <Link
              to={`/videos/${id}`}
              className="mb-2 inline-flex items-center gap-2 text-xs text-slate-500 hover:text-slate-800 print:hidden"
            >
              <ArrowLeft size={14} />
              Back to Case
            </Link>
            <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
              Forensic Report
            </div>
            <h1 className="mt-1 text-2xl font-semibold">
              Firinne Dhigiteach - Video Evidence Assessment
            </h1>
            <p className="mt-1 text-slate-500">
              Case #{id} | Generated {new Date(result.created_at).toUTCString()}
            </p>
          </div>

          <button
            onClick={handleGeneratePdf}
            className="rounded border border-slate-300 px-4 py-2 text-sm print:hidden"
          >
            Print / Save PDF
          </button>
        </div>

        <Section title="Case Metadata">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <KV label="Filename" value={result.filename} />
            <KV
              label="Classification"
              value={formatClassification(result.classification)}
            />
            <KV
              label="Forensic Score"
              value={result.forensic_score.toFixed(3)}
            />
            <KV label="Frame Count" value={String(result.frame_count)} />
          </div>
        </Section>

        <Section title="Integrity Snapshot">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <KV label="SHA-256" value={formatUnknown(hashes?.sha256)} mono />
            <KV label="MD5" value={formatUnknown(hashes?.md5)} mono />
          </div>
          <div className="mt-4 text-sm text-slate-700">
            Classification is based on sampled frame analysis.
          </div>
        </Section>

        <Section title="Extracted Audio Summary">
          {!audio ? (
            <p className="text-sm text-slate-700">
              No extracted-audio analysis was stored for this case.
            </p>
          ) : (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <KV label="Available" value={audio.available ? "Yes" : "No"} />
              <KV
                label="Classification"
                value={formatClassification(audio.classification)}
              />
              <KV
                label="Forensic Score"
                value={
                  typeof audio.forensic_score === "number"
                    ? audio.forensic_score.toFixed(3)
                    : "Unavailable"
                }
              />
              <KV
                label="Hashes Match"
                value={
                  typeof audioHashes?.hashes_match === "boolean"
                    ? audioHashes.hashes_match
                      ? "Yes"
                      : "No"
                    : "Unavailable"
                }
              />
            </div>
          )}
        </Section>

        {audio ? (
          <>
            {audio.error ? (
              <Section title="Audio Processing Note">
                <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                  {audio.error}
                </div>
              </Section>
            ) : null}

            <Section title="Audio Signal Metrics">
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <KV
                  label="Duration"
                  value={formatUnknown(audioMeta?.duration_seconds, "seconds")}
                />
                <KV
                  label="Sample Rate"
                  value={formatUnknown(audioMeta?.sample_rate_hz, "Hz")}
                />
                <KV
                  label="Channels"
                  value={formatUnknown(audioMeta?.channels)}
                />
                <KV
                  label="Peak Level"
                  value={formatUnknown(audioFeatures?.peak_level)}
                />
                <KV
                  label="Dynamic Range"
                  value={formatUnknown(audioFeatures?.dynamic_range_db, "dB")}
                />
                <KV
                  label="Zero Crossing Rate"
                  value={formatUnknown(audioFeatures?.zero_crossing_rate)}
                />
                <KV
                  label="Repetition Score"
                  value={formatUnknown(audioFeatures?.repetition_score)}
                />
                <KV
                  label="Spectral Flatness"
                  value={formatUnknown(audioFeatures?.spectral_flatness)}
                />
                <KV
                  label="Transcoded"
                  value={formatUnknown(audioFeatures?.transcoded_for_analysis)}
                />
                <KV
                  label="FFmpeg Error"
                  value={formatUnknown(audioFeatures?.ffmpeg_transcode_error)}
                />
                {audioSegments ? (
                  <>
                    <KV
                      label="Segment Size"
                      value={formatUnknown(
                        audioSegments.segment_duration_seconds,
                        "seconds",
                      )}
                    />
                    <KV
                      label="Segment Count"
                      value={formatUnknown(audioSegments.segment_count)}
                    />
                    <KV
                      label="Segment RMS Std"
                      value={formatNestedStat(
                        audioSegments,
                        "rms_level",
                        "std",
                      )}
                    />
                    <KV
                      label="Segment ZCR Std"
                      value={formatNestedStat(
                        audioSegments,
                        "zero_crossing_rate",
                        "std",
                      )}
                    />
                  </>
                ) : null}
              </div>
            </Section>

            <Section title="Audio Exhibits">
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {audio.waveform_path ? (
                  <figure
                    className="rounded-lg border border-slate-200 p-3"
                    style={{ breakInside: "avoid" }}
                  >
                    <figcaption className="mb-2 text-xs uppercase text-slate-500">
                      Exhibit - Extracted Audio Waveform
                    </figcaption>
                    <img
                      src={fixPath(audio.waveform_path)}
                      alt="Extracted audio waveform"
                      className="w-full object-contain"
                    />
                  </figure>
                ) : null}

                {typeof audioFeatures?.spectrogram_path === "string" ? (
                  <figure
                    className="rounded-lg border border-slate-200 p-3"
                    style={{ breakInside: "avoid" }}
                  >
                    <figcaption className="mb-2 text-xs uppercase text-slate-500">
                      Exhibit - Extracted Audio Spectrogram
                    </figcaption>
                    <img
                      src={fixPath(audioFeatures.spectrogram_path as string)}
                      alt="Extracted audio spectrogram"
                      className="w-full object-contain"
                    />
                  </figure>
                ) : null}
              </div>
            </Section>

            <Section title="Audio Integrity">
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <KV
                  label="SHA-256"
                  value={
                    audioHashes?.hashes_after?.sha256 ||
                    audioHashes?.hashes_before?.sha256 ||
                    "Unavailable"
                  }
                  mono
                />
                <KV
                  label="MD5"
                  value={
                    audioHashes?.hashes_after?.md5 ||
                    audioHashes?.hashes_before?.md5 ||
                    "Unavailable"
                  }
                  mono
                />
              </div>
            </Section>
          </>
        ) : null}

        <PrintDecisionSummary
          verdict={formatClassification(result.classification)}
          scoreValue={result.forensic_score.toFixed(3)}
          rationale={[
            `Frame sample count: ${result.frame_count}`,
            "Primary decision is based on sampled frame analysis.",
            audio?.available
              ? `Extracted audio classification: ${formatClassification(
                  audio.classification,
                )}`
              : "No extracted-audio evidence was available.",
            integrityNote,
          ]}
          note="Visual and extracted-audio evidence streams should be interpreted together rather than as independent final determinations."
        />

        {representativeFrame ? (
          <Section title="Selected Frame Forensic Exhibits">
            <div className="mb-3 text-sm text-slate-700">
              Frame {representativeFrame.frame_index} at{" "}
              {representativeFrame.timestamp_sec.toFixed(2)}s{" · "}
              Score {representativeFrame.forensic_score.toFixed(3)}
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              {representativeFrame.saved_path ? (
                <figure
                  className="rounded-lg border border-slate-200 p-3"
                  style={{ breakInside: "avoid" }}
                >
                  <figcaption className="mb-2 text-xs uppercase text-slate-500">
                    Exhibit A - Original Frame
                  </figcaption>
                  <img
                    src={fixPath(representativeFrame.saved_path)}
                    alt={`Frame ${representativeFrame.frame_index} original`}
                    className="max-h-[320px] w-full object-contain"
                  />
                </figure>
              ) : null}

              {representativeFrame.gradcam_heatmap ? (
                <figure
                  className="rounded-lg border border-slate-200 p-3"
                  style={{ breakInside: "avoid" }}
                >
                  <figcaption className="mb-2 text-xs uppercase text-slate-500">
                    Exhibit B - GradCAM
                  </figcaption>
                  <img
                    src={fixPath(representativeFrame.gradcam_heatmap)}
                    alt={`Frame ${representativeFrame.frame_index} GradCAM`}
                    className="max-h-[320px] w-full object-contain"
                  />
                </figure>
              ) : null}

              {representativeFrame.ela_heatmap ? (
                <figure
                  className="rounded-lg border border-slate-200 p-3"
                  style={{ breakInside: "avoid" }}
                >
                  <figcaption className="mb-2 text-xs uppercase text-slate-500">
                    Exhibit C - ELA
                  </figcaption>
                  <img
                    src={fixPath(representativeFrame.ela_heatmap)}
                    alt={`Frame ${representativeFrame.frame_index} ELA`}
                    className="max-h-[320px] w-full object-contain"
                  />
                </figure>
              ) : null}

              {representativeFrame.noise_residual?.noise_heatmap_path ? (
                <figure
                  className="rounded-lg border border-slate-200 p-3"
                  style={{ breakInside: "avoid" }}
                >
                  <figcaption className="mb-2 text-xs uppercase text-slate-500">
                    Exhibit D - Noise Residual
                  </figcaption>
                  <img
                    src={fixPath(
                      representativeFrame.noise_residual.noise_heatmap_path,
                    )}
                    alt={`Frame ${representativeFrame.frame_index} noise residual`}
                    className="max-h-[320px] w-full object-contain"
                  />
                </figure>
              ) : null}

              {representativeFrame.jpeg_qtables?.double_compression
                ?.jpeg_quality_heatmap_path ? (
                <figure
                  className="rounded-lg border border-slate-200 p-3"
                  style={{ breakInside: "avoid" }}
                >
                  <figcaption className="mb-2 text-xs uppercase text-slate-500">
                    Exhibit E - JPEG Quality Heatmap
                  </figcaption>
                  <img
                    src={fixPath(
                      representativeFrame.jpeg_qtables.double_compression
                        .jpeg_quality_heatmap_path,
                    )}
                    alt={`Frame ${representativeFrame.frame_index} JPEG quality heatmap`}
                    className="max-h-[320px] w-full object-contain"
                  />
                </figure>
              ) : null}
            </div>
          </Section>
        ) : null}

        <Section title="Sample Frame Exhibits">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {frames.map((frame) => (
              <figure
                key={frame.frame_index}
                className="overflow-hidden rounded-lg border border-slate-200"
                style={{ breakInside: "avoid" }}
              >
                <img
                  src={fixPath(frame.saved_path)}
                  alt={`Frame ${frame.frame_index}`}
                  className="h-28 w-full object-cover"
                />
                <figcaption className="p-2 text-xs text-slate-700">
                  Frame {frame.frame_index} | {frame.timestamp_sec.toFixed(2)}s
                </figcaption>
              </figure>
            ))}
          </div>
        </Section>

        <PrintAppliedSettings settings={result.applied_settings} />

        <section className="border-t border-slate-200 pt-3 text-[11px] text-slate-500">
          This report is decision-support evidence and should be interpreted
          with contextual forensic review.
        </section>
      </div>
    </div>
  );
};
