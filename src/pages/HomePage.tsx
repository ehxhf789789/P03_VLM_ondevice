import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "../i18n";
import { useAnalysis } from "../hooks/useAnalysis";
import { ConditionGradeBadge } from "../components/ConditionGradeBadge";
import type { AnalysisResult } from "../types/analysis";

export function HomePage() {
  const { t, locale, setLocale } = useTranslation();
  const navigate = useNavigate();
  const {
    modelStatus,
    isDownloading,
    downloadProgress,
    error,
    history,
    refreshModelStatus,
    downloadModel,
    loadModel,
    loadHistory,
  } = useAnalysis();

  useEffect(() => {
    refreshModelStatus();
    loadHistory();
  }, [refreshModelStatus, loadHistory]);

  const statusText = (() => {
    if (typeof modelStatus === "string") {
      const key = {
        NotDownloaded: "notDownloaded",
        Downloaded: "downloaded",
        Loading: "loading",
        Ready: "ready",
      }[modelStatus] as keyof typeof t.home;
      return t.home[key] ?? modelStatus;
    }
    if ("Downloading" in modelStatus) return t.home.downloading;
    if ("Error" in modelStatus) return `${t.home.error}: ${modelStatus.Error}`;
    return "";
  })();

  const statusColor = (() => {
    if (modelStatus === "Ready") return "text-green-600";
    if (modelStatus === "Downloaded") return "text-blue-600";
    if (typeof modelStatus === "object" && "Error" in modelStatus)
      return "text-red-600";
    return "text-gray-600";
  })();

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="px-4 pt-12 pb-4">
        <div className="flex items-center justify-between mb-1">
          <h1 className="text-2xl font-bold text-gray-900">{t.app.title}</h1>
          <button
            onClick={() => setLocale(locale === "ko" ? "en" : "ko")}
            className="px-3 py-1 text-sm bg-gray-100 rounded-full"
          >
            {locale === "ko" ? "EN" : "한국어"}
          </button>
        </div>
        <p className="text-sm text-gray-500">{t.app.subtitle}</p>
      </header>

      {/* Model Status Card */}
      <div className="px-4 mb-4">
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-gray-500">
              {t.home.modelStatus}
            </span>
            <span className={`text-sm font-semibold ${statusColor}`}>
              {statusText}
            </span>
          </div>

          {/* Download progress */}
          {isDownloading && downloadProgress && (
            <div className="mb-3">
              <div className="flex justify-between text-xs text-gray-500 mb-1">
                <span>{downloadProgress.file_name}</span>
                <span>
                  {downloadProgress.current_file}/{downloadProgress.total_files}
                </span>
              </div>
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-500 rounded-full transition-all"
                  style={{ width: `${downloadProgress.progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Action buttons */}
          <div className="flex gap-2">
            {modelStatus === "NotDownloaded" && (
              <button
                onClick={downloadModel}
                disabled={isDownloading}
                className="btn-primary flex-1"
              >
                {t.home.downloadModel}
              </button>
            )}
            {modelStatus === "Downloaded" && (
              <button onClick={loadModel} className="btn-primary flex-1">
                {t.home.loadModel}
              </button>
            )}
            {modelStatus === "Ready" && (
              <button
                onClick={() => navigate("/capture")}
                className="btn-primary flex-1"
              >
                {t.home.startInspection}
              </button>
            )}
            {typeof modelStatus === "object" && "Error" in modelStatus && (
              <button onClick={refreshModelStatus} className="btn-secondary flex-1">
                {t.common.confirm}
              </button>
            )}
          </div>

          {error && (
            <p className="text-xs text-red-500 mt-2">{error}</p>
          )}
        </div>
      </div>

      {/* History */}
      <div className="flex-1 px-4 overflow-y-auto pb-8">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">
          {t.home.history}
        </h2>
        {history.length === 0 ? (
          <p className="text-sm text-gray-400 text-center py-8">
            {t.home.noHistory}
          </p>
        ) : (
          <div className="space-y-2">
            {history.map((item: AnalysisResult) => (
              <button
                key={item.id}
                onClick={() =>
                  navigate("/result", { state: { result: item } })
                }
                className="card w-full text-left flex items-center gap-3"
              >
                <ConditionGradeBadge grade={item.condition_grade} size="sm" />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-900 truncate">
                    {item.item_name}
                  </p>
                  <p className="text-xs text-gray-500">{item.item_category}</p>
                </div>
                <span className="text-sm text-gray-400">
                  {item.condition_score}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
