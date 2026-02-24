import type { AnalysisEvent } from "../types/analysis";
import { useTranslation } from "../i18n";

interface Props {
  event: AnalysisEvent | null;
}

export function AnalysisProgress({ event }: Props) {
  const { t } = useTranslation();
  const progress = event?.progress ?? 0;
  const stage = event?.stage ?? "loading";

  const stageMessage =
    t.analysis[stage as keyof typeof t.analysis] ?? event?.message ?? "";

  return (
    <div className="flex flex-col items-center gap-6 py-12">
      {/* Spinner */}
      <div className="relative w-24 h-24">
        <svg className="w-full h-full animate-spin" viewBox="0 0 100 100">
          <circle
            cx="50"
            cy="50"
            r="42"
            fill="none"
            stroke="#e5e7eb"
            strokeWidth="8"
          />
          <circle
            cx="50"
            cy="50"
            r="42"
            fill="none"
            stroke="#6366f1"
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${progress * 2.64} ${264 - progress * 2.64}`}
            strokeDashoffset="0"
            transform="rotate(-90 50 50)"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-lg font-bold text-indigo-600">
            {Math.round(progress)}%
          </span>
        </div>
      </div>

      <div className="text-center">
        <h3 className="text-lg font-semibold text-gray-900">
          {t.analysis.title}
        </h3>
        <p className="text-sm text-gray-500 mt-1">{stageMessage}</p>
      </div>

      {/* Progress bar */}
      <div className="w-full max-w-xs">
        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-indigo-500 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>
    </div>
  );
}
