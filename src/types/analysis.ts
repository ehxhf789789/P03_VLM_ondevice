/** 제품 상태 등급 */
export type ConditionGrade = "S" | "A" | "B" | "C" | "D";

/** 손상 유형 */
export type DamageType =
  | "Rust"
  | "Scratch"
  | "Dent"
  | "Discoloration"
  | "Crack"
  | "Corrosion"
  | "Wear"
  | "Stain"
  | "Chip"
  | "Peeling"
  | { Other: string };

/** 손상 심각도 */
export type Severity = "Minor" | "Moderate" | "Severe";

/** 개별 손상 정보 */
export interface DamageInfo {
  damage_type: DamageType;
  severity: Severity;
  location: string;
  description: string;
  confidence: number;
}

/** 분석 결과 */
export interface AnalysisResult {
  id: string;
  timestamp: string;
  item_name: string;
  item_category: string;
  is_new: boolean;
  condition_grade: ConditionGrade;
  condition_score: number;
  damages: DamageInfo[];
  overall_description: string;
  grade_reasoning: string;
}

/** 모델 상태 */
export type ModelStatus =
  | "NotDownloaded"
  | { Downloading: { progress: number } }
  | "Downloaded"
  | "Loading"
  | "Ready"
  | { Error: string };

/** 다운로드 이벤트 */
export interface DownloadEvent {
  file_name: string;
  progress: number;
  total_files: number;
  current_file: number;
}

/** 분석 이벤트 */
export interface AnalysisEvent {
  stage: string;
  progress: number;
  message: string;
}

/** DamageType에서 키 문자열 추출 */
export function getDamageTypeKey(dt: DamageType): string {
  if (typeof dt === "string") return dt.toLowerCase();
  return "other";
}

/** DamageType 표시 이름 추출 */
export function getDamageTypeLabel(dt: DamageType): string {
  if (typeof dt === "string") return dt;
  return dt.Other;
}
