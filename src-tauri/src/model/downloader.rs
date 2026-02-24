use futures_util::StreamExt;
use std::path::{Path, PathBuf};
use tauri::ipc::Channel;

use crate::analysis::DownloadEvent;

/// 다운로드할 모델 파일 목록
const MODEL_FILES: &[(&str, &str)] = &[
    (
        "vision_encoder.onnx",
        "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/onnx/vision_encoder.onnx",
    ),
    (
        "embed_tokens.onnx",
        "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/onnx/embed_tokens.onnx",
    ),
    (
        "decoder_model_merged.onnx",
        "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/onnx/decoder_model_merged.onnx",
    ),
    (
        "tokenizer.json",
        "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/tokenizer.json",
    ),
];

/// 모델 디렉토리에서 모든 필수 파일이 존재하는지 확인한다.
pub fn all_model_files_exist(model_dir: &Path) -> bool {
    MODEL_FILES
        .iter()
        .all(|(name, _)| model_dir.join(name).exists())
}

/// 모델 파일 이름 목록을 반환한다.
pub fn model_file_names() -> Vec<&'static str> {
    MODEL_FILES.iter().map(|(name, _)| *name).collect()
}

/// 모델 파일을 HuggingFace에서 다운로드한다.
pub async fn download_models(
    model_dir: &Path,
    channel: Channel<DownloadEvent>,
) -> Result<(), String> {
    std::fs::create_dir_all(model_dir)
        .map_err(|e| format!("Failed to create model directory: {}", e))?;

    let total_files = MODEL_FILES.len();
    let client = reqwest::Client::new();

    for (idx, (file_name, url)) in MODEL_FILES.iter().enumerate() {
        let dest_path = model_dir.join(file_name);

        // 이미 존재하면 스킵
        if dest_path.exists() {
            let _ = channel.send(DownloadEvent {
                file_name: file_name.to_string(),
                progress: 100.0,
                total_files,
                current_file: idx + 1,
            });
            continue;
        }

        download_file(&client, url, &dest_path, file_name, idx, total_files, &channel)
            .await?;
    }

    Ok(())
}

async fn download_file(
    client: &reqwest::Client,
    url: &str,
    dest: &PathBuf,
    file_name: &str,
    idx: usize,
    total_files: usize,
    channel: &Channel<DownloadEvent>,
) -> Result<(), String> {
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Download request failed for {}: {}", file_name, e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Download failed for {}: HTTP {}",
            file_name,
            response.status()
        ));
    }

    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;

    let temp_path = dest.with_extension("part");
    let mut file = tokio::fs::File::create(&temp_path)
        .await
        .map_err(|e| format!("Failed to create file {}: {}", file_name, e))?;

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download error for {}: {}", file_name, e))?;
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
            .await
            .map_err(|e| format!("Write error for {}: {}", file_name, e))?;

        downloaded += chunk.len() as u64;
        let progress = if total_size > 0 {
            (downloaded as f64 / total_size as f64) * 100.0
        } else {
            0.0
        };

        let _ = channel.send(DownloadEvent {
            file_name: file_name.to_string(),
            progress,
            total_files,
            current_file: idx + 1,
        });
    }

    // 완료 후 임시 파일을 최종 경로로 이동
    tokio::fs::rename(&temp_path, dest)
        .await
        .map_err(|e| format!("Failed to rename {}: {}", file_name, e))?;

    Ok(())
}
