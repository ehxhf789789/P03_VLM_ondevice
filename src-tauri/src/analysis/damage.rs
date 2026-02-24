use super::types::*;

/// VLM 텍스트 출력에서 JSON 블록을 추출한다.
pub fn extract_json_from_text(text: &str) -> Option<String> {
    // ```json ... ``` 블록 우선 탐색
    if let Some(start) = text.find("```json") {
        let json_start = start + 7;
        if let Some(end) = text[json_start..].find("```") {
            return Some(text[json_start..json_start + end].trim().to_string());
        }
    }

    // ``` ... ``` 블록 탐색
    if let Some(start) = text.find("```") {
        let json_start = start + 3;
        if let Some(end) = text[json_start..].find("```") {
            let candidate = text[json_start..json_start + end].trim();
            if candidate.starts_with('{') {
                return Some(candidate.to_string());
            }
        }
    }

    // 첫 번째 { ... 마지막 } 추출
    let first_brace = text.find('{')?;
    let last_brace = text.rfind('}')?;
    if first_brace < last_brace {
        return Some(text[first_brace..=last_brace].to_string());
    }

    None
}

/// VLM 원시 출력을 파싱하여 구조화된 데이터로 변환한다.
pub fn parse_vlm_output(raw_text: &str) -> Result<(String, String, bool, String, Vec<DamageInfo>), String> {
    let json_str = extract_json_from_text(raw_text)
        .ok_or_else(|| "No JSON found in VLM output".to_string())?;

    let raw: VlmRawOutput = serde_json::from_str(&json_str)
        .map_err(|e| format!("JSON parse error: {}", e))?;

    let item_name = raw.item_name.unwrap_or_else(|| "Unknown Item".to_string());
    let item_category = raw.item_category.unwrap_or_else(|| "Unknown".to_string());
    let is_new = raw.is_new.unwrap_or(false);
    let overall_description = raw.overall_description.unwrap_or_default();

    let damages: Vec<DamageInfo> = raw
        .damages
        .unwrap_or_default()
        .into_iter()
        .map(|d| DamageInfo {
            damage_type: DamageType::from_str_loose(
                &d.damage_type.unwrap_or_else(|| "other".to_string()),
            ),
            severity: Severity::from_str_loose(
                &d.severity.unwrap_or_else(|| "moderate".to_string()),
            ),
            location: d.location.unwrap_or_else(|| "unspecified".to_string()),
            description: d.description.unwrap_or_default(),
            confidence: d.confidence.unwrap_or(0.5).clamp(0.0, 1.0),
        })
        .collect();

    Ok((item_name, item_category, is_new, overall_description, damages))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_from_markdown() {
        let text = r#"Here is the analysis:
```json
{"item_name": "Laptop", "damages": []}
```
Done."#;
        let json = extract_json_from_text(text).unwrap();
        assert!(json.contains("Laptop"));
    }

    #[test]
    fn test_extract_json_bare() {
        let text = r#"Result: {"item_name": "Phone"} end"#;
        let json = extract_json_from_text(text).unwrap();
        assert!(json.contains("Phone"));
    }

    #[test]
    fn test_parse_vlm_output() {
        let raw = r#"{"item_name":"Wrench","item_category":"Tool","is_new":false,"overall_description":"A used wrench with visible rust","damages":[{"damage_type":"rust","severity":"moderate","location":"handle","description":"Surface rust on handle","confidence":0.85}]}"#;
        let (name, cat, is_new, desc, damages) = parse_vlm_output(raw).unwrap();
        assert_eq!(name, "Wrench");
        assert_eq!(cat, "Tool");
        assert!(!is_new);
        assert!(!desc.is_empty());
        assert_eq!(damages.len(), 1);
        assert_eq!(damages[0].damage_type, DamageType::Rust);
    }
}
