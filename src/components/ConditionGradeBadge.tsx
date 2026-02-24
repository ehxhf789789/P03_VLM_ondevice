import type { ConditionGrade } from "../types/analysis";
import { useTranslation } from "../i18n";

const gradeColors: Record<ConditionGrade, string> = {
  S: "bg-grade-S text-white",
  A: "bg-grade-A text-white",
  B: "bg-grade-B text-white",
  C: "bg-grade-C text-white",
  D: "bg-grade-D text-white",
};

const gradeSizes = {
  sm: "w-8 h-8 text-sm",
  md: "w-12 h-12 text-xl",
  lg: "w-20 h-20 text-3xl",
};

interface Props {
  grade: ConditionGrade;
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
}

export function ConditionGradeBadge({ grade, size = "md", showLabel = false }: Props) {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col items-center gap-1">
      <div
        className={`${gradeColors[grade]} ${gradeSizes[size]} rounded-full flex items-center justify-center font-bold shadow-lg`}
      >
        {grade}
      </div>
      {showLabel && (
        <span className="text-xs text-gray-500">
          {t.grade[grade]}
        </span>
      )}
    </div>
  );
}
