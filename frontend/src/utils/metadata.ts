const DEFAULT_MAX_STRING_LENGTH = 300;

const shouldOmitKey = (key: string) => {
  const normalized = key.toLowerCase();
  return normalized.includes("jpegthumbnail");
};

const describeOmission = (label: string, length?: number) => {
  if (typeof length === "number") {
    return `[omitted ${label} (${length} chars)]`;
  }
  return `[omitted ${label}]`;
};

export const sanitizeMetadata = (
  value: unknown,
  maxStringLength: number = DEFAULT_MAX_STRING_LENGTH
): unknown => {
  if (typeof value === "string") {
    if (value.length > maxStringLength) {
      return describeOmission("long string", value.length);
    }
    return value;
  }

  if (Array.isArray(value)) {
    return value.map((item) => sanitizeMetadata(item, maxStringLength));
  }

  if (value && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const sanitized: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(obj)) {
      if (shouldOmitKey(key)) {
        sanitized[key] =
          typeof entry === "string"
            ? describeOmission("JPEG thumbnail data", entry.length)
            : describeOmission("JPEG thumbnail data");
        continue;
      }
      sanitized[key] = sanitizeMetadata(entry, maxStringLength);
    }
    return sanitized;
  }

  return value;
};
