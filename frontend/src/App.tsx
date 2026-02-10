import React from "react";
import { HashRouter, Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "./components/Layout";
import { UploadPage } from "./pages/UploadPage";
import { HistoryPage } from "./pages/HistoryPage";
import { DetailPage } from "./pages/DetailPage";
import { VideoDetailPage } from "./pages/VideoDetailPage";
import { PrintDetailPage } from "./pages/PrintDetailPage";
import { PrintVideoPage } from "./pages/PrintVideoPage";
import { AuditLogPage } from "./pages/AuditLogPage";
import { SettingsPage } from "./pages/SettingsPage";

const App: React.FC = () => {
  return (
    <HashRouter>
      <Routes>
        <Route path="/print/records/:id" element={<PrintDetailPage />} />
        <Route path="/print/videos/:id" element={<PrintVideoPage />} />
        <Route path="/" element={<Layout />}>
          <Route index element={<UploadPage />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="audit" element={<AuditLogPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="records/:id" element={<DetailPage />} />
          <Route path="videos/:id" element={<VideoDetailPage />} />

          {/* Redirect unknown routes */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </HashRouter>
  );
};

export default App;
