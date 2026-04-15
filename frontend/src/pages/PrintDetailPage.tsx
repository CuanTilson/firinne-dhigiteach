import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getRecordById } from "../services/api";
import type { AnalysisResult } from "../types";
import { fixPath } from "../constants";
import { sanitizeMetadata } from "../utils/metadata";
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

const PrintProvenance = ({
  detector,
  fusionMode,
}: {
  detector?: Record<string, unknown> | null;
  fusionMode?: string | null;
}) => (
  <Section title="Analysis Provenance">
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      <KV
        label="Detector"
        value={formatUnknown(detector?.display_name ?? detector?.name)}
      />
      <KV
        label="Model Version"
        value={formatUnknown(detector?.model_version)}
      />
      <KV
        label="Dataset Version"
        value={formatUnknown(detector?.dataset_version)}
      />
      <KV label="Fusion Mode" value={formatUnknown(fusionMode)} />
      <KV
        label="Weights SHA-256"
        value={formatUnknown(asRecord(detector?.weights)?.sha256)}
        mono
      />
    </div>
  </Section>
);

const PrintC2PA = ({ c2pa }: { c2pa: AnalysisResult["c2pa"] }) => {
  const hasC2PA = Boolean(c2pa?.has_c2pa);
  const signatureValid = c2pa?.signature_valid === true;
  const aiAssertions = c2pa?.ai_assertions_found?.length ?? 0;

  const summary = !hasC2PA
    ? "No C2PA manifest detected."
    : !signatureValid
      ? "C2PA manifest present, but signature could not be validated."
      : aiAssertions > 0
        ? "Valid C2PA manifest with AI-related assertions."
        : "Valid C2PA manifest without explicit AI-related assertions.";

  return (
    <Section title="C2PA Provenance Summary">
      <div className="mb-4 text-sm text-slate-700">{summary}</div>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <KV label="Manifest Present" value={hasC2PA ? "Yes" : "No"} />
        <KV
          label="Signature Valid"
          value={
            c2pa?.signature_valid === true
              ? "Yes"
              : c2pa?.signature_valid === false
                ? "No"
                : "Unavailable"
          }
        />
        <KV label="AI Assertions" value={String(aiAssertions)} />
        <KV
          label="Claim Generator"
          value={formatUnknown(c2pa?.claim_generator)}
        />
      </div>
    </Section>
  );
};

export const PrintDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<AnalysisResult | null>(null);
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
        const data = await getRecordById(Number(id));
        setResult(data);
      } catch {
        setError("Could not load analysis details.");
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

  const safeMetadata = sanitizeMetadata(result.raw_metadata ?? {});
  const sha256 =
    result.file_integrity?.hashes_after?.sha256 ||
    result.file_integrity?.hashes_before?.sha256 ||
    result.file_integrity?.hashes?.sha256 ||
    "Not available";
  const md5 =
    result.file_integrity?.hashes_after?.md5 ||
    result.file_integrity?.hashes_before?.md5 ||
    result.file_integrity?.hashes?.md5 ||
    "Not available";

  const findings = result.metadata_anomalies?.findings ?? [];
  const mlProbability =
    typeof result.ml_prediction?.probability === "number"
      ? result.ml_prediction.probability.toFixed(3)
      : "Unavailable";

  const detector =
    result.ml_prediction?.detector &&
    typeof result.ml_prediction.detector === "object"
      ? (result.ml_prediction.detector as Record<string, unknown>)
      : null;

  const fusionMode =
    result.forensic_score_json &&
    typeof result.forensic_score_json === "object" &&
    typeof result.forensic_score_json.provenance === "object"
      ? String(
          (result.forensic_score_json.provenance as Record<string, unknown>)
            .fusion_mode || "",
        )
      : null;

  const qualityBadges: string[] = [];
  if (result.c2pa?.has_c2pa) qualityBadges.push("C2PA Present");
  if (result.ai_watermark?.stable_diffusion_detected) {
    qualityBadges.push("Watermark Signal Detected");
  }
  if ((result.jpeg_qtables?.inconsistency_score ?? 0) > 0.15) {
    qualityBadges.push("JPEG Recompression Signal");
  }

  return (
    <div className="min-h-screen bg-slate-100 px-4 py-8 text-slate-900 print:bg-white print:p-0">
      <div className="mx-auto max-w-5xl space-y-6 rounded-xl border border-slate-200 bg-white p-6 print:border-0 print:p-0">
        <div
          className="flex items-start justify-between border-b border-slate-200 pb-4"
          style={{ breakInside: "avoid" }}
        >
          <div>
            <Link
              to={`/records/${id}`}
              className="mb-2 inline-flex items-center gap-2 text-xs text-slate-500 hover:text-slate-800 print:hidden"
            >
              <ArrowLeft size={14} />
              Back to Case
            </Link>
            <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
              Forensic Report
            </div>
            <h1 className="mt-1 text-2xl font-semibold">
              Firinne Dhigiteach - Image Evidence Assessment
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
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            {qualityBadges.length === 0 ? (
              <span className="rounded border border-slate-300 px-2 py-1 text-xs text-slate-600">
                No dominant quality flags
              </span>
            ) : (
              qualityBadges.map((badge) => (
                <span
                  key={badge}
                  className="rounded border border-slate-300 px-2 py-1 text-xs text-slate-700"
                >
                  {badge}
                </span>
              ))
            )}
          </div>
        </Section>

        <Section title="Integrity Snapshot">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <KV label="SHA-256" value={sha256} mono />
            <KV label="MD5" value={md5} mono />
            <KV
              label="JPEG Structure"
              value={
                result.file_integrity?.jpeg_structure?.valid_jpeg
                  ? "Valid"
                  : "Issues Detected"
              }
            />
            <KV
              label="Hash Integrity"
              value={
                result.file_integrity?.hashes_match === false
                  ? "Mismatch detected"
                  : "No hash drift detected"
              }
            />
          </div>
        </Section>

        <Section title="Analyst Findings">
          {findings.length === 0 ? (
            <p className="text-slate-700">
              No suspicious metadata patterns flagged.
            </p>
          ) : (
            <ul className="list-disc space-y-1 pl-5 text-slate-800">
              {findings.slice(0, 8).map((item, index) => (
                <li key={`${item}-${index}`}>{item}</li>
              ))}
            </ul>
          )}
        </Section>

        <PrintDecisionSummary
          verdict={formatClassification(result.classification)}
          scoreValue={result.forensic_score.toFixed(3)}
          rationale={[
            `ML probability: ${mlProbability}`,
            `Metadata findings flagged: ${findings.length}`,
            result.c2pa?.has_c2pa
              ? "C2PA provenance is present."
              : "No C2PA provenance was detected.",
            qualityBadges[0] || "No dominant quality flags were raised.",
          ]}
          note="The current decision reflects the present scoring pipeline and should be reviewed alongside the exhibits and metadata appendix."
        />

        <PrintProvenance detector={detector} fusionMode={fusionMode} />

        <PrintC2PA c2pa={result.c2pa} />

        <Section title="Visual Exhibits">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {result.saved_path ? (
              <figure
                className="rounded-lg border border-slate-200 p-3"
                style={{ breakInside: "avoid" }}
              >
                <figcaption className="mb-2 text-xs uppercase text-slate-500">
                  Exhibit A - Original
                </figcaption>
                <img
                  src={fixPath(result.saved_path)}
                  alt="Original"
                  className="max-h-[320px] w-full object-contain"
                />
              </figure>
            ) : null}

            {result.gradcam_heatmap ? (
              <figure
                className="rounded-lg border border-slate-200 p-3"
                style={{ breakInside: "avoid" }}
              >
                <figcaption className="mb-2 text-xs uppercase text-slate-500">
                  Exhibit B - GradCAM
                </figcaption>
                <img
                  src={fixPath(result.gradcam_heatmap)}
                  alt="GradCAM"
                  className="max-h-[320px] w-full object-contain"
                />
              </figure>
            ) : null}

            {result.ela_heatmap ? (
              <figure
                className="rounded-lg border border-slate-200 p-3"
                style={{ breakInside: "avoid" }}
              >
                <figcaption className="mb-2 text-xs uppercase text-slate-500">
                  Exhibit C - ELA
                </figcaption>
                <img
                  src={fixPath(result.ela_heatmap)}
                  alt="ELA"
                  className="max-h-[320px] w-full object-contain"
                />
              </figure>
            ) : null}

            {result.noise_residual?.noise_heatmap_path ? (
              <figure
                className="rounded-lg border border-slate-200 p-3"
                style={{ breakInside: "avoid" }}
              >
                <figcaption className="mb-2 text-xs uppercase text-slate-500">
                  Exhibit D - Noise Residual
                </figcaption>
                <img
                  src={fixPath(result.noise_residual.noise_heatmap_path)}
                  alt="Noise"
                  className="max-h-[320px] w-full object-contain"
                />
              </figure>
            ) : null}
          </div>
        </Section>

        <Section title="Metadata Appendix">
          <div className="text-sm text-slate-700">
            Full sanitised metadata is appended in the PDF appendix.
          </div>
          <pre data-pdf-append-text className="hidden print:block">
            {JSON.stringify(safeMetadata, null, 2)}
          </pre>
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
