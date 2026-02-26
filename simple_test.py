"""간단한 SmolVLM 테스트"""
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import urllib.request
import io

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)
print("Model loaded!")

# 샘플 이미지 다운로드 (손상된 자동차 이미지)
print("\nDownloading test image...")
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Single.jpg/800px-Banana-Single.jpg"
with urllib.request.urlopen(url) as response:
    image_data = response.read()
image = Image.open(io.BytesIO(image_data)).convert("RGB")
image.save("test_banana.jpg")
print("Test image saved as test_banana.jpg")

# 테스트 프롬프트들
test_prompts = [
    # 1. 기본 인식
    "What is this object? Describe it.",

    # 2. 상태 분석
    """Analyze the condition of this object. Check for any damage, defects, or wear.
Output JSON format:
{
  "object": "name",
  "condition": "new/good/fair/poor",
  "damages": [{"type": "", "location": "", "severity": ""}]
}""",

    # 3. 바운딩 박스 테스트
    """Can you detect objects and provide bounding box coordinates?
If yes, output: {"objects": [{"name": "", "bbox": [x1, y1, x2, y2]}]}
If no, just describe the object location."""
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'='*60}")
    print(f"Test {i}")
    print(f"{'='*60}")
    print(f"Prompt: {prompt[:100]}...")

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    response = processor.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()

    print(f"\nResponse:\n{response}")
