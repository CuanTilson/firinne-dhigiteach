import React from "react";
import { HistoryPage } from "./HistoryPage";

export const AudioHistoryPage: React.FC = () => (
  <HistoryPage
    initialMediaType="audio"
    title="Audio Analysis History"
    description="Archive of standalone audio investigations."
  />
);
