import { invoke, Channel } from "@tauri-apps/api/core";
import type {
  ModelStatus,
  AnalysisResult,
  DownloadEvent,
  AnalysisEvent,
} from "../types/analysis";

/** 모델 상태 확인 */
export async function getModelStatus(): Promise<ModelStatus> {
  return invoke<ModelStatus>("get_model_status");
}

/** 모델 다운로드 (진행률 콜백) */
export async function downloadModel(
  onProgress: (event: DownloadEvent) => void
): Promise<void> {
  const channel = new Channel<DownloadEvent>();
  channel.onmessage = onProgress;
  return invoke("download_model", { onProgress: channel });
}

/** 모델 로드 */
export async function loadModel(): Promise<void> {
  return invoke("load_model");
}

/** 이미지 분석 (진행률 콜백) */
export async function analyzeImage(
  imagePath: string,
  onProgress: (event: AnalysisEvent) => void
): Promise<AnalysisResult> {
  const channel = new Channel<AnalysisEvent>();
  channel.onmessage = onProgress;
  return invoke<AnalysisResult>("analyze_image", {
    imagePath,
    onProgress: channel,
  });
}

/** 분석 히스토리 조회 */
export async function getHistory(): Promise<AnalysisResult[]> {
  return invoke<AnalysisResult[]>("get_history");
}
