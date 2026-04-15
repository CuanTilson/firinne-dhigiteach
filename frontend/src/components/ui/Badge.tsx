import React from "react";
import type { ClassificationType } from "../../types";
import { CLASSIFICATION_COLORS, CLASSIFICATION_LABELS } from "../../constants";

interface BadgeProps {
  type: ClassificationType;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({ type, className = "" }) => {
  const styles = CLASSIFICATION_COLORS[type];
  const label = CLASSIFICATION_LABELS[type];

  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-1 text-sm font-semibold ${styles} ${className}`}
    >
      {label}
    </span>
  );
};
