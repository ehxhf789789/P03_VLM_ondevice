use serde::{Deserialize, Serialize};

/// 제품 상태 등급 (S가 최고, D가 최저)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConditionGrade {
    S, // 90~100: 거의 새 것
    A, // 75~89: 양호
    B, // 55~74: 보통
    C, // 35~54: 불량
    D, // 0~34: 심각
}

impl ConditionGrade {
    pub fn from_score(score: u32) -> Self {
        match score {
            90..=100 => Self::S,
            75..=89 => Self::A,
            55..=74 => Self::B,
            35..=54 => Self::C,
            _ => Self::D,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::S => "S",
            Self::A => "A",
            Self::B => "B",
            Self::C => "C",
            Self::D => "D",
        }
    }
}

/// 손상 유형 (10종 + Other)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum DamageType {
    Rust,
    Scratch,
    Dent,
    Discoloration,
    Crack,
    Corrosion,
    Wear,
    Stain,
    Chip,
    Peeling,
    Other(String),
}

impl DamageType {
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().trim() {
            "rust" => Self::Rust,
            "scratch" | "scratches" => Self::Scratch,
            "dent" | "dents" | "dented" => Self::Dent,
            "discoloration" | "discolored" | "fading" => Self::Discoloration,
            "crack" | "cracks" | "cracked" => Self::Crack,
            "corrosion" | "corroded" => Self::Corrosion,
            "wear" | "worn" | "abrasion" => Self::Wear,
            "stain" | "stains" | "stained" => Self::Stain,
            "chip" | "chips" | "chipped" | "chipping" => Self::Chip,
            "peeling" | "peeled" | "flaking" => Self::Peeling,
            other => Self::Other(other.to_string()),
        }
    }

    pub fn base_weight(&self) -> f64 {
        match self {
            Self::Crack => 15.0,
            Self::Corrosion => 14.0,
            Self::Rust => 12.0,
            Self::Chip => 11.0,
            Self::Dent => 10.0,
            Self::Peeling => 9.0,
            Self::Scratch => 7.0,
            Self::Discoloration => 5.0,
            Self::Wear => 6.0,
            Self::Stain => 4.0,
            Self::Other(_) => 8.0,
        }
    }
}

/// 손상 심각도
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Minor,
    Moderate,
    Severe,
}

impl Severity {
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().trim() {
            "minor" | "light" | "slight" => Self::Minor,
            "moderate" | "medium" => Self::Moderate,
            "severe" | "heavy" | "critical" | "major" => Self::Severe,
            _ => Self::Moderate,
        }
    }

    pub fn multiplier(&self) -> f64 {
        match self {
            Self::Minor => 0.5,
            Self::Moderate => 1.0,
            Self::Severe => 1.8,
        }
    }
}

/// 개별 손상 정보
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DamageInfo {
    pub damage_type: DamageType,
    pub severity: Severity,
    pub location: String,
    pub description: String,
    pub confidence: f64,
}

/// 분석 결과
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub id: String,
    pub timestamp: String,
    pub item_name: String,
    pub item_category: String,
    pub is_new: bool,
    pub condition_grade: ConditionGrade,
    pub condition_score: u32,
    pub damages: Vec<DamageInfo>,
    pub overall_description: String,
    pub grade_reasoning: String,
}

/// 모델 상태
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    NotDownloaded,
    Downloading { progress: f64 },
    Downloaded,
    Loading,
    Ready,
    Error(String),
}

/// 다운로드 이벤트 (Channel 전달용)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadEvent {
    pub file_name: String,
    pub progress: f64,
    pub total_files: usize,
    pub current_file: usize,
}

/// 분석 이벤트 (Channel 전달용)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisEvent {
    pub stage: String,
    pub progress: f64,
    pub message: String,
}

/// VLM에서 파싱할 원시 JSON 구조
#[derive(Debug, Deserialize)]
pub struct VlmRawOutput {
    pub item_name: Option<String>,
    pub item_category: Option<String>,
    pub is_new: Option<bool>,
    pub overall_description: Option<String>,
    pub damages: Option<Vec<VlmRawDamage>>,
}

#[derive(Debug, Deserialize)]
pub struct VlmRawDamage {
    pub damage_type: Option<String>,
    pub severity: Option<String>,
    pub location: Option<String>,
    pub description: Option<String>,
    pub confidence: Option<f64>,
}
