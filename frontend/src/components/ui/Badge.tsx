import React from "react";
import type { ClassificationType } from "../../types";
import { CLASSIFICATION_COLORS, CLASSIFICATION_LABELS } from "../../constants";

interface BadgeProps {
  type: ClassificationType;
}

export const Badge: React.FC<BadgeProps> = ({ type }) => {
  const styles = CLASSIFICATION_COLORS[type];
  const label = CLASSIFICATION_LABELS[type];

  return (
    <span
      className={`px-3 py-1 rounded-full text-sm font-semibold border ${styles}`}
    >
      {label}
    </span>
  );
};
