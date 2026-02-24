use std::path::PathBuf;
use std::sync::Mutex;

use crate::analysis::AnalysisResult;
use crate::model::SmolVlmEngine;

/// 앱 전역 상태
pub struct AppState {
    pub engine: Mutex<Option<SmolVlmEngine>>,
    pub history: Mutex<Vec<AnalysisResult>>,
    pub model_dir: PathBuf,
}

impl AppState {
    pub fn new(model_dir: PathBuf) -> Self {
        Self {
            engine: Mutex::new(None),
            history: Mutex::new(Vec::new()),
            model_dir,
        }
    }
}
