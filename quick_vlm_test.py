"""빠른 VLM 성능 테스트"""
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import numpy as np

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)
print("Model loaded!")

# 손상된 표면 이미지 생성
print("\nCreating test image with damage...")
img = np.ones((384, 384, 3), dtype=np.uint8) * 170
# 녹 (우측 상단)
img[50:130, 250:350] = [140, 85, 40]
# 스크래치 (중앙 대각선)
for i in range(100, 280):
    img[i, i-30:i-27] = [50, 45, 40]
# 찌그러짐 (좌측 하단)
for y in range(260, 340):
    for x in range(60, 140):
        dist = np.sqrt((x-100)**2 + (y-300)**2)
        if dist < 40:
            img[y, x] = [int(170 - (40-dist)*1.5)] * 3

image = Image.fromarray(img)
image.save("test_damage.png")
print("Test image saved!")

def test(prompt, name):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=250, do_sample=False,
                                  pad_token_id=processor.tokenizer.pad_token_id)

    full = processor.decode(outputs[0], skip_special_tokens=False)
    if "Assistant:" in full:
        resp = full.split("Assistant:")[-1]
        if "<end_of_utterance>" in resp:
            resp = resp.split("<end_of_utterance>")[0]
    else:
        resp = full[-400:]

    print(f"Response:\n{resp.strip()}")
    return resp.strip()

# 테스트 1: 기본 인식
test("What do you see in this image? Describe it.", "Basic Recognition")

# 테스트 2: 손상 감지
test("""Look at this surface image carefully.
Is there any damage? If yes, describe:
1. Type of damage
2. Location (top/bottom/left/right/center)
3. Severity (minor/moderate/severe)""", "Damage Detection")

# 테스트 3: JSON 출력
resp = test("""Analyze this surface for damage.
Output ONLY JSON format:
{"damages": [{"type": "name", "location": "position", "severity": "level"}]}""", "JSON Output")

# JSON 파싱 시도
print("\n--- JSON Parsing Test ---")
try:
    import json
    if "{" in resp:
        json_str = resp[resp.find("{"):resp.rfind("}")+1]
        parsed = json.loads(json_str)
        print(f"SUCCESS: Parsed JSON = {parsed}")
except Exception as e:
    print(f"FAILED: {e}")

# 테스트 4: 위치 설명
test("""Describe the location of any visible marks or damage.
Use grid position: top-left, top-center, top-right, center-left, center, center-right, bottom-left, bottom-center, bottom-right""", "Location Description")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
