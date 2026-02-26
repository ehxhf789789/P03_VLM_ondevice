"""
SmolVLM 모델 성능 테스트 스크립트
- 손상 수준 분석
- 손상 위치 및 범위 (바운딩 박스)
- 객체 인식
- 객체 종류 추론
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import json
import sys

# 모델 로드
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
print(f"Model loaded on: {model.device}")

def test_capability(image_path: str, prompt: str, description: str):
    """특정 기능을 테스트하고 결과를 출력"""
    print(f"\n{'='*60}")
    print(f"테스트: {description}")
    print(f"{'='*60}")

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt")

    if torch.cuda.is_available():
        inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    response = processor.decode(outputs[0], skip_special_tokens=True)
    # 프롬프트 부분 제거
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()

    print(f"\n프롬프트:\n{prompt[:200]}...")
    print(f"\n응답:\n{response}")
    return response

def run_all_tests(image_path: str):
    """모든 테스트 실행"""

    # 1. 기본 객체 인식
    test_capability(
        image_path,
        "What object is shown in this image? Describe it briefly.",
        "기본 객체 인식"
    )

    # 2. 상세 객체 분류
    test_capability(
        image_path,
        """Identify the object in this image and classify it.
Output JSON:
{
  "object_name": "specific name",
  "category": "main category",
  "subcategory": "subcategory",
  "material": "material type",
  "color": "main color"
}""",
        "상세 객체 분류"
    )

    # 3. 손상 감지 및 위치
    test_capability(
        image_path,
        """Analyze this image for any damage or defects.
For each damage found, provide:
- Type of damage (scratch, dent, rust, crack, etc.)
- Severity (minor, moderate, severe)
- Location in the image (describe position: top-left, center, bottom-right, etc.)
- Approximate size/extent

Output as JSON:
{
  "damages": [
    {
      "type": "damage type",
      "severity": "level",
      "location": "position description",
      "extent": "size or percentage affected"
    }
  ]
}""",
        "손상 감지 및 위치 분석"
    )

    # 4. 바운딩 박스 테스트
    test_capability(
        image_path,
        """Detect all visible damages in this image and provide their bounding box coordinates.
Use normalized coordinates (0-1) for the bounding boxes.

Output JSON:
{
  "detections": [
    {
      "label": "damage type",
      "bbox": [x_min, y_min, x_max, y_max],
      "confidence": 0.0-1.0
    }
  ]
}

If you cannot provide exact coordinates, describe the relative position.""",
        "바운딩 박스 좌표 추출"
    )

    # 5. 전체 상태 평가
    test_capability(
        image_path,
        """Perform a comprehensive product condition assessment.

Evaluate:
1. Product identification (what is it?)
2. Overall condition grade (A: Like New, B: Good, C: Fair, D: Poor, F: Damaged)
3. List all visible damages with severity
4. Estimated remaining lifespan or usability
5. Recommended actions (keep, repair, replace)

Output detailed JSON:
{
  "product": {
    "name": "",
    "type": "",
    "brand": "if visible"
  },
  "condition": {
    "grade": "A-F",
    "score": 0-100,
    "summary": ""
  },
  "damages": [],
  "recommendation": ""
}""",
        "종합 상태 평가"
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_vlm_capability.py <image_path>")
        print("\n테스트 이미지 없이 간단한 모델 테스트 실행...")

        # 테스트 이미지 생성
        import numpy as np
        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        test_img.save("test_random.png")

        test_capability(
            "test_random.png",
            "Describe what you see in this image.",
            "모델 동작 확인"
        )
    else:
        run_all_tests(sys.argv[1])
