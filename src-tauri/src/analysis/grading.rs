use super::types::*;

/// 손상 목록으로부터 상태 점수(0~100)를 계산한다.
/// 점수가 높을수록 상태가 좋다 (손상이 적다).
pub fn calculate_condition_score(damages: &[DamageInfo]) -> u32 {
    if damages.is_empty() {
        return 95; // 손상이 없으면 거의 새 것
    }

    let total_penalty: f64 = damages
        .iter()
        .map(|d| {
            let base = d.damage_type.base_weight();
            let severity_mult = d.severity.multiplier();
            let confidence_factor = 0.3 + 0.7 * d.confidence;
            base * severity_mult * confidence_factor
        })
        .sum();

    // 100에서 감점, 최소 0
    let score = (100.0 - total_penalty).max(0.0).min(100.0);
    score.round() as u32
}

/// 점수 → 등급 변환
pub fn determine_grade(score: u32) -> ConditionGrade {
    ConditionGrade::from_score(score)
}

/// 등급에 대한 사유 설명 생성
pub fn generate_grade_reasoning(
    grade: ConditionGrade,
    score: u32,
    damages: &[DamageInfo],
) -> String {
    let damage_count = damages.len();
    let severe_count = damages
        .iter()
        .filter(|d| d.severity == Severity::Severe)
        .count();

    match grade {
        ConditionGrade::S => format!(
            "Score {}/100. Product is in excellent condition with minimal to no visible damage ({} issue(s) detected).",
            score, damage_count
        ),
        ConditionGrade::A => format!(
            "Score {}/100. Product is in good condition with minor cosmetic issues ({} issue(s) detected).",
            score, damage_count
        ),
        ConditionGrade::B => format!(
            "Score {}/100. Product shows moderate wear with noticeable damage ({} issue(s) detected, {} severe).",
            score, damage_count, severe_count
        ),
        ConditionGrade::C => format!(
            "Score {}/100. Product has significant damage affecting usability ({} issue(s) detected, {} severe).",
            score, damage_count, severe_count
        ),
        ConditionGrade::D => format!(
            "Score {}/100. Product is in poor condition with extensive damage ({} issue(s) detected, {} severe). May not be functional.",
            score, damage_count, severe_count
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_damages_high_score() {
        let score = calculate_condition_score(&[]);
        assert!(score >= 90);
        assert_eq!(determine_grade(score), ConditionGrade::S);
    }

    #[test]
    fn test_minor_scratch() {
        let damages = vec![DamageInfo {
            damage_type: DamageType::Scratch,
            severity: Severity::Minor,
            location: "surface".to_string(),
            description: "Light scratch".to_string(),
            confidence: 0.8,
        }];
        let score = calculate_condition_score(&damages);
        assert!(score >= 75, "Score {} should be >= 75", score);
    }

    #[test]
    fn test_severe_crack_low_score() {
        let damages = vec![
            DamageInfo {
                damage_type: DamageType::Crack,
                severity: Severity::Severe,
                location: "body".to_string(),
                description: "Deep crack".to_string(),
                confidence: 0.95,
            },
            DamageInfo {
                damage_type: DamageType::Corrosion,
                severity: Severity::Severe,
                location: "base".to_string(),
                description: "Heavy corrosion".to_string(),
                confidence: 0.9,
            },
            DamageInfo {
                damage_type: DamageType::Dent,
                severity: Severity::Moderate,
                location: "side".to_string(),
                description: "Large dent".to_string(),
                confidence: 0.85,
            },
        ];
        let score = calculate_condition_score(&damages);
        assert!(score < 55, "Score {} should be < 55", score);
    }
}
