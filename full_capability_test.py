"""SmolVLM 전체 성능 테스트"""
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import urllib.request
import io

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)
print("Model loaded!\n")

def run_test(image, prompt, test_name):
    """단일 테스트 실행"""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    print(f"Prompt: {prompt[:150]}...")

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    full_response = processor.decode(outputs[0], skip_special_tokens=False)
    # Assistant: 이후만 추출
    if "Assistant:" in full_response:
        response = full_response.split("Assistant:")[-1].strip()
        if "<end_of_utterance>" in response:
            response = response.split("<end_of_utterance>")[0].strip()
    else:
        response = full_response

    print(f"\nRESPONSE:\n{response}")
    return response

# 테스트 이미지 다운로드
print("Downloading test image (scratched surface)...")
# 손상된 금속 표면 이미지
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Rust_and_dirt.jpg/640px-Rust_and_dirt.jpg"
try:
    with urllib.request.urlopen(url) as response:
        image_data = response.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    print("Image downloaded!")
except:
    print("Failed to download, creating synthetic damaged surface...")
    import numpy as np
    # 손상 시뮬레이션 이미지 생성
    img = np.ones((224, 224, 3), dtype=np.uint8) * 180  # 회색 표면
    # 스크래치 추가
    for i in range(20, 200):
        img[i, i:i+3] = [80, 60, 40]  # 대각선 스크래치
        img[i, 100:105] = [100, 70, 50]  # 세로 스크래치
    # 녹 추가
    img[50:100, 150:200] = [139, 90, 43]  # 녹 패치
    image = Image.fromarray(img)

# ============================================
# 테스트 1: 기본 객체 인식
# ============================================
run_test(image,
    "What object or surface is shown in this image? Describe it briefly.",
    "Basic Object Recognition"
)

# ============================================
# 테스트 2: 손상 감지
# ============================================
run_test(image,
    """Analyze this image for damage or defects.
List any visible damage including:
- Type of damage (rust, scratch, dent, crack, etc.)
- Severity (minor, moderate, severe)
- Location description""",
    "Damage Detection"
)

# ============================================
# 테스트 3: JSON 출력
# ============================================
run_test(image,
    """Analyze this image and output ONLY valid JSON:
{
  "object": "what is shown",
  "condition": "new/good/fair/poor/damaged",
  "damages": [{"type": "", "severity": "", "location": ""}]
}""",
    "Structured JSON Output"
)

# ============================================
# 테스트 4: 바운딩 박스/좌표
# ============================================
run_test(image,
    """Detect damaged areas in this image.
For each damage, provide approximate location using:
- Position: top-left, top-center, top-right, center-left, center, center-right, bottom-left, bottom-center, bottom-right
- Or coordinates as percentage: x% from left, y% from top

Output format: {"damages": [{"type": "", "position": "", "size": ""}]}""",
    "Location/Coordinate Detection"
)

# ============================================
# 테스트 5: 상세 분류
# ============================================
run_test(image,
    """Provide detailed classification:
1. Material type (metal, plastic, wood, fabric, etc.)
2. Surface finish (smooth, textured, painted, etc.)
3. Estimated age/wear level
4. Quality grade (A=excellent, B=good, C=fair, D=poor, F=damaged)

Output as JSON.""",
    "Detailed Classification"
)

# ============================================
# 테스트 6: 종합 평가
# ============================================
run_test(image,
    """Perform comprehensive product condition assessment:
{
  "product_info": {"type": "", "material": "", "color": ""},
  "condition": {"grade": "A-F", "score": 0-100},
  "damages": [{"type": "", "severity": "", "location": "", "description": ""}],
  "recommendation": "keep/repair/replace",
  "notes": ""
}
Output ONLY the JSON, no other text.""",
    "Comprehensive Assessment"
)

print("\n" + "="*70)
print("ALL TESTS COMPLETED")
print("="*70)
