import { useNavigate, useLocation } from "react-router-dom";
import { useTranslation } from "../i18n";
import type { AnalysisResult } from "../types/analysis";
import { ConditionGradeBadge } from "../components/ConditionGradeBadge";
import { DamageBreakdown } from "../components/DamageBreakdown";

interface LocationState {
  result: AnalysisResult;
  previewSrc?: string;
}

export function ResultPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const { result, previewSrc } = (location.state as LocationState) ?? {};

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <p className="text-gray-500">No results available</p>
        <button onClick={() => navigate("/")} className="btn-primary mt-4">
          {t.result.backToHome}
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="px-4 pt-12 pb-4 flex items-center gap-3">
        <button
          onClick={() => navigate("/")}
          className="w-10 h-10 flex items-center justify-center rounded-full bg-gray-100"
        >
          ←
        </button>
        <h1 className="text-xl font-bold text-gray-900">{t.result.title}</h1>
      </header>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto px-4 pb-8 space-y-4">
        {/* Grade hero */}
        <div className="card flex items-center gap-4">
          {previewSrc && (
            <img
              src={previewSrc}
              alt={result.item_name}
              className="w-20 h-20 rounded-xl object-cover"
            />
          )}
          <div className="flex-1">
            <h2 className="text-lg font-bold text-gray-900">
              {result.item_name}
            </h2>
            <p className="text-sm text-gray-500">{result.item_category}</p>
            {result.is_new && (
              <span className="inline-block mt-1 px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded-full">
                {t.result.newItem}
              </span>
            )}
          </div>
          <ConditionGradeBadge
            grade={result.condition_grade}
            size="lg"
            showLabel
          />
        </div>

        {/* Score */}
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-500">
              {t.result.conditionScore}
            </span>
            <span className="text-2xl font-bold text-gray-900">
              {result.condition_score}
              <span className="text-sm text-gray-400 font-normal">/100</span>
            </span>
          </div>
          <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${result.condition_score}%`,
                backgroundColor:
                  result.condition_score >= 90
                    ? "#6366f1"
                    : result.condition_score >= 75
                      ? "#22c55e"
                      : result.condition_score >= 55
                        ? "#eab308"
                        : result.condition_score >= 35
                          ? "#f97316"
                          : "#ef4444",
              }}
            />
          </div>
        </div>

        {/* Grade reasoning */}
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500 mb-1">
            {t.result.gradeReasoning}
          </h3>
          <p className="text-sm text-gray-700">{result.grade_reasoning}</p>
        </div>

        {/* Overall description */}
        {result.overall_description && (
          <div className="card">
            <h3 className="text-sm font-medium text-gray-500 mb-1">
              {t.result.overallDescription}
            </h3>
            <p className="text-sm text-gray-700">
              {result.overall_description}
            </p>
          </div>
        )}

        {/* Damage breakdown */}
        <DamageBreakdown damages={result.damages} />

        {/* Action buttons */}
        <div className="space-y-2 pt-2">
          <button
            onClick={() => navigate("/capture")}
            className="btn-primary w-full"
          >
            {t.result.analyzeAnother}
          </button>
          <button
            onClick={() => navigate("/")}
            className="btn-secondary w-full"
          >
            {t.result.backToHome}
          </button>
        </div>
      </div>
    </div>
  );
}
