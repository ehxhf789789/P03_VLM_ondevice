import { useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAnalysis } from "../hooks/useAnalysis";
import { AnalysisProgress } from "../components/AnalysisProgress";

interface LocationState {
  imagePath: string;
  previewSrc: string;
}

export function AnalysisPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { imagePath, previewSrc } = (location.state as LocationState) ?? {};
  const { isAnalyzing, analysisProgress, analyzeImage, error } = useAnalysis();

  useEffect(() => {
    if (!imagePath) {
      navigate("/capture");
      return;
    }

    let cancelled = false;
    (async () => {
      const result = await analyzeImage(imagePath);
      if (!cancelled && result) {
        navigate("/result", { state: { result, previewSrc }, replace: true });
      }
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imagePath]);

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-screen px-4">
        <div className="text-4xl mb-4">⚠</div>
        <p className="text-red-600 text-center mb-4">{error}</p>
        <button onClick={() => navigate("/")} className="btn-secondary">
          Back
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center h-screen px-4">
      {previewSrc && (
        <img
          src={previewSrc}
          alt="Analyzing"
          className="w-32 h-32 rounded-2xl object-cover mb-6 opacity-60"
        />
      )}
      <AnalysisProgress event={isAnalyzing ? analysisProgress : null} />
    </div>
  );
}
