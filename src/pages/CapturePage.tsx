import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { open } from "@tauri-apps/plugin-dialog";
import { useTranslation } from "../i18n";
import { ImagePreview } from "../components/ImagePreview";
import { convertFileSrc } from "@tauri-apps/api/core";

export function CapturePage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [imagePath, setImagePath] = useState<string | null>(null);
  const [previewSrc, setPreviewSrc] = useState<string | null>(null);

  const handleChooseFromGallery = async () => {
    const file = await open({
      multiple: false,
      filters: [
        {
          name: "Images",
          extensions: ["png", "jpg", "jpeg", "webp", "bmp"],
        },
      ],
    });
    if (file) {
      setImagePath(file);
      setPreviewSrc(convertFileSrc(file));
    }
  };

  const handleAnalyze = () => {
    if (imagePath) {
      navigate("/analysis", { state: { imagePath, previewSrc } });
    }
  };

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
        <h1 className="text-xl font-bold text-gray-900">{t.capture.title}</h1>
      </header>

      {/* Image Preview Area */}
      <div className="flex-1 px-4 flex flex-col items-center justify-center">
        {previewSrc ? (
          <div className="w-full max-w-sm">
            <ImagePreview
              src={previewSrc}
              alt="Selected"
              className="aspect-square mb-4"
            />
            <p className="text-sm text-center text-gray-500">
              {t.capture.selectedImage}
            </p>
          </div>
        ) : (
          <div className="w-full max-w-sm aspect-square rounded-2xl border-2 border-dashed border-gray-300 flex items-center justify-center">
            <p className="text-gray-400 text-sm">{t.capture.title}</p>
          </div>
        )}
      </div>

      {/* Buttons */}
      <div className="px-4 pb-8 space-y-3">
        <button
          onClick={handleChooseFromGallery}
          className="btn-secondary w-full"
        >
          {t.capture.chooseFromGallery}
        </button>
        <button
          onClick={handleAnalyze}
          disabled={!imagePath}
          className="btn-primary w-full"
        >
          {t.capture.analyze}
        </button>
      </div>
    </div>
  );
}
