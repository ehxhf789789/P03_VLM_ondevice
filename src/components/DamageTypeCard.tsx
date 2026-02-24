import type { DamageInfo } from "../types/analysis";
import { getDamageTypeKey, getDamageTypeLabel } from "../types/analysis";
import { useTranslation } from "../i18n";

const severityColors = {
  Minor: "bg-green-100 text-green-800",
  Moderate: "bg-yellow-100 text-yellow-800",
  Severe: "bg-red-100 text-red-800",
};

interface Props {
  damage: DamageInfo;
  index: number;
}

export function DamageTypeCard({ damage, index }: Props) {
  const { t } = useTranslation();
  const typeKey = getDamageTypeKey(damage.damage_type);
  const typeLabel =
    (t.damage as Record<string, string>)[typeKey] ??
    getDamageTypeLabel(damage.damage_type);
  const severityLabel =
    t.severity[damage.severity.toLowerCase() as keyof typeof t.severity] ??
    damage.severity;

  return (
    <div className="card">
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-400">
            #{index + 1}
          </span>
          <h4 className="font-semibold text-gray-900">{typeLabel}</h4>
        </div>
        <span
          className={`px-2 py-0.5 rounded-full text-xs font-medium ${severityColors[damage.severity]}`}
        >
          {severityLabel}
        </span>
      </div>
      <div className="space-y-1 text-sm">
        <div className="flex gap-2">
          <span className="text-gray-500 shrink-0">{t.result.location}:</span>
          <span className="text-gray-700">{damage.location}</span>
        </div>
        <div className="flex gap-2">
          <span className="text-gray-500 shrink-0">
            {t.result.description}:
          </span>
          <span className="text-gray-700">{damage.description}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-gray-500">{t.result.confidence}:</span>
          <div className="flex-1 h-1.5 bg-gray-200 rounded-full">
            <div
              className="h-full bg-indigo-500 rounded-full"
              style={{ width: `${damage.confidence * 100}%` }}
            />
          </div>
          <span className="text-xs text-gray-500">
            {Math.round(damage.confidence * 100)}%
          </span>
        </div>
      </div>
    </div>
  );
}
