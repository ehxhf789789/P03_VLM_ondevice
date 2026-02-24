import type { DamageInfo } from "../types/analysis";
import { useTranslation } from "../i18n";
import { DamageTypeCard } from "./DamageTypeCard";

interface Props {
  damages: DamageInfo[];
}

export function DamageBreakdown({ damages }: Props) {
  const { t } = useTranslation();

  if (damages.length === 0) {
    return (
      <div className="card text-center py-8">
        <div className="text-4xl mb-2">✓</div>
        <p className="text-gray-500">{t.result.noDamage}</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <h3 className="font-semibold text-gray-900">
        {t.result.damageReport} ({damages.length})
      </h3>
      {damages.map((damage, i) => (
        <DamageTypeCard key={i} damage={damage} index={i} />
      ))}
    </div>
  );
}
