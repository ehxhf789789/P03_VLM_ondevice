import { useState, useCallback } from "react";
import type {
  AnalysisResult,
  AnalysisEvent,
  ModelStatus,
  DownloadEvent,
} from "../types/analysis";
import * as api from "../lib/tauri";

interface AnalysisState {
  modelStatus: ModelStatus;
  isDownloading: boolean;
  downloadProgress: DownloadEvent | null;
  isAnalyzing: boolean;
  analysisProgress: AnalysisEvent | null;
  result: AnalysisResult | null;
  error: string | null;
  history: AnalysisResult[];
}

export function useAnalysis() {
  const [state, setState] = useState<AnalysisState>({
    modelStatus: "NotDownloaded",
    isDownloading: false,
    downloadProgress: null,
    isAnalyzing: false,
    analysisProgress: null,
    result: null,
    error: null,
    history: [],
  });

  const refreshModelStatus = useCallback(async () => {
    try {
      const status = await api.getModelStatus();
      setState((s) => ({ ...s, modelStatus: status, error: null }));
    } catch (e) {
      setState((s) => ({ ...s, error: String(e) }));
    }
  }, []);

  const downloadModel = useCallback(async () => {
    setState((s) => ({
      ...s,
      isDownloading: true,
      error: null,
      modelStatus: { Downloading: { progress: 0 } },
    }));
    try {
      await api.downloadModel((event) => {
        setState((s) => ({
          ...s,
          downloadProgress: event,
          modelStatus: {
            Downloading: {
              progress:
                ((event.current_file - 1) / event.total_files +
                  event.progress / 100 / event.total_files) *
                100,
            },
          },
        }));
      });
      setState((s) => ({
        ...s,
        isDownloading: false,
        modelStatus: "Downloaded",
        downloadProgress: null,
      }));
    } catch (e) {
      setState((s) => ({
        ...s,
        isDownloading: false,
        modelStatus: { Error: String(e) },
        error: String(e),
      }));
    }
  }, []);

  const loadModel = useCallback(async () => {
    setState((s) => ({ ...s, modelStatus: "Loading", error: null }));
    try {
      await api.loadModel();
      setState((s) => ({ ...s, modelStatus: "Ready" }));
    } catch (e) {
      setState((s) => ({
        ...s,
        modelStatus: { Error: String(e) },
        error: String(e),
      }));
    }
  }, []);

  const analyzeImage = useCallback(async (imagePath: string) => {
    setState((s) => ({
      ...s,
      isAnalyzing: true,
      result: null,
      error: null,
      analysisProgress: null,
    }));
    try {
      const result = await api.analyzeImage(imagePath, (event) => {
        setState((s) => ({ ...s, analysisProgress: event }));
      });
      setState((s) => ({
        ...s,
        isAnalyzing: false,
        result,
        analysisProgress: null,
      }));
      return result;
    } catch (e) {
      setState((s) => ({
        ...s,
        isAnalyzing: false,
        error: String(e),
        analysisProgress: null,
      }));
      return null;
    }
  }, []);

  const loadHistory = useCallback(async () => {
    try {
      const history = await api.getHistory();
      setState((s) => ({ ...s, history }));
    } catch (e) {
      setState((s) => ({ ...s, error: String(e) }));
    }
  }, []);

  const clearResult = useCallback(() => {
    setState((s) => ({ ...s, result: null, error: null }));
  }, []);

  return {
    ...state,
    refreshModelStatus,
    downloadModel,
    loadModel,
    analyzeImage,
    loadHistory,
    clearResult,
  };
}
