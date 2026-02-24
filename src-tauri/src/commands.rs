use std::fs;
use tauri::ipc::Channel;
use tauri::State;

use crate::analysis::damage::parse_vlm_output;
use crate::analysis::grading::{calculate_condition_score, determine_grade, generate_grade_reasoning};
use crate::analysis::types::*;
use crate::model::downloader;
use crate::model::prompts;
use crate::model::SmolVlmEngine;
use crate::state::AppState;

/// 모델 다운로드 상태를 확인한다.
#[tauri::command]
pub fn get_model_status(state: State<'_, AppState>) -> ModelStatus {
    let engine = state.engine.lock().unwrap();
    if engine.is_some() {
        ModelStatus::Ready
    } else if downloader::all_model_files_exist(&state.model_dir) {
        ModelStatus::Downloaded
    } else {
        ModelStatus::NotDownloaded
    }
}

/// 모델 파일을 HuggingFace에서 다운로드한다.
#[tauri::command]
pub async fn download_model(
    state: State<'_, AppState>,
    on_progress: Channel<DownloadEvent>,
) -> Result<(), String> {
    downloader::download_models(&state.model_dir, on_progress).await
}

/// 다운로드된 모델을 메모리에 로드한다.
#[tauri::command]
pub async fn load_model(state: State<'_, AppState>) -> Result<(), String> {
    if !downloader::all_model_files_exist(&state.model_dir) {
        return Err("Model files not found. Please download first.".to_string());
    }

    let model_dir = state.model_dir.clone();
    let engine = tokio::task::spawn_blocking(move || SmolVlmEngine::load(&model_dir))
        .await
        .map_err(|e| format!("Task join error: {}", e))??;

    let mut engine_lock = state.engine.lock().unwrap();
    *engine_lock = Some(engine);

    Ok(())
}

/// 이미지를 분석하여 결과를 반환한다.
#[tauri::command]
pub async fn analyze_image(
    state: State<'_, AppState>,
    image_path: String,
    on_progress: Channel<AnalysisEvent>,
) -> Result<AnalysisResult, String> {
    // 1. 진행 상태: 이미지 로드
    let _ = on_progress.send(AnalysisEvent {
        stage: "loading".to_string(),
        progress: 10.0,
        message: "Loading image...".to_string(),
    });

    // 이미지 파일 읽기
    let image_bytes = fs::read(&image_path)
        .map_err(|e| format!("Failed to read image file: {}", e))?;

    // 2. 진행 상태: 추론 시작
    let _ = on_progress.send(AnalysisEvent {
        stage: "analyzing".to_string(),
        progress: 30.0,
        message: "Running VLM analysis...".to_string(),
    });

    // 엔진 잠금 및 추론 실행
    let mut engine_lock = state.engine.lock().unwrap();
    let engine = engine_lock
        .as_mut()
        .ok_or_else(|| "Model not loaded".to_string())?;

    let prompt = prompts::build_chat_prompt(true);
    let raw_output = engine.run_inference(&image_bytes, &prompt)?;

    // 3. 진행 상태: 결과 파싱
    let _ = on_progress.send(AnalysisEvent {
        stage: "parsing".to_string(),
        progress: 80.0,
        message: "Parsing results...".to_string(),
    });

    // VLM 출력 파싱
    let (item_name, item_category, is_new, overall_description, damages) =
        parse_vlm_output(&raw_output)?;

    // 점수 및 등급 계산
    let condition_score = calculate_condition_score(&damages);
    let condition_grade = determine_grade(condition_score);
    let grade_reasoning =
        generate_grade_reasoning(condition_grade, condition_score, &damages);

    let result = AnalysisResult {
        id: uuid::Uuid::new_v4().to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        item_name,
        item_category,
        is_new,
        condition_grade,
        condition_score,
        damages,
        overall_description,
        grade_reasoning,
    };

    // 히스토리에 추가
    let mut history = state.history.lock().unwrap();
    history.push(result.clone());

    // 4. 완료
    let _ = on_progress.send(AnalysisEvent {
        stage: "complete".to_string(),
        progress: 100.0,
        message: "Analysis complete!".to_string(),
    });

    Ok(result)
}

/// 분석 히스토리를 반환한다.
#[tauri::command]
pub fn get_history(state: State<'_, AppState>) -> Vec<AnalysisResult> {
    let history = state.history.lock().unwrap();
    history.clone()
}
