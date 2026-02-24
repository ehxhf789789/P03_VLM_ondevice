/// VLM에게 제품 상태 분석을 요청하는 시스템 프롬프트
pub const ANALYSIS_SYSTEM_PROMPT: &str = r#"You are an expert product condition inspector. Analyze the provided product image and output a JSON object with the following structure. Be thorough and precise.

Damage types to check: rust, scratch, dent, discoloration, crack, corrosion, wear, stain, chip, peeling.
Severity levels: minor, moderate, severe.

Output ONLY valid JSON, no other text:"#;

/// 사용자 프롬프트 (이미지와 함께 전달됨)
pub const ANALYSIS_USER_PROMPT: &str = r#"Analyze this product image. Identify the product, assess its condition, and list all visible damage.

Respond with ONLY this JSON format:
```json
{
  "item_name": "product name",
  "item_category": "category (e.g., Electronics, Tool, Furniture, Clothing, Vehicle Part, Other)",
  "is_new": false,
  "overall_description": "Brief overall condition description",
  "damages": [
    {
      "damage_type": "scratch",
      "severity": "minor",
      "location": "top surface",
      "description": "Light surface scratch approximately 3cm long",
      "confidence": 0.85
    }
  ]
}
```

If no damage is found, return an empty damages array. Be specific about locations and descriptions."#;

/// 채팅 템플릿 형식으로 프롬프트를 조립한다.
pub fn build_chat_prompt(has_image: bool) -> String {
    let mut prompt = String::new();

    // SmolVLM 채팅 형식
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str(ANALYSIS_SYSTEM_PROMPT);
    prompt.push_str("<|im_end|>\n");

    prompt.push_str("<|im_start|>user\n");
    if has_image {
        prompt.push_str("<image>\n");
    }
    prompt.push_str(ANALYSIS_USER_PROMPT);
    prompt.push_str("<|im_end|>\n");

    prompt.push_str("<|im_start|>assistant\n");

    prompt
}
