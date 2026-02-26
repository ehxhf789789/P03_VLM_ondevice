"""최소한의 SmolVLM 테스트"""
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import numpy as np

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

print("1. Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)
print("   Model loaded!")

# 테스트 이미지 생성 (간단한 빨간 사각형)
print("\n2. Creating test image...")
img = np.zeros((224, 224, 3), dtype=np.uint8)
img[50:150, 50:150] = [255, 0, 0]  # 빨간 사각형
image = Image.fromarray(img)
print("   Test image created!")

# 테스트 1: 기본 묘사
print("\n3. Running inference...")
prompt = "Describe what you see in this image."
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt_text, images=[image], return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

response = processor.decode(outputs[0], skip_special_tokens=True)
print(f"\n=== TEST 1: Basic Description ===")
print(f"Response: {response}")

# 테스트 2: JSON 출력
print("\n4. Testing JSON output...")
prompt2 = """What do you see? Output JSON:
{"object": "name", "color": "color", "position": "location"}"""

messages2 = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt2}]}]
prompt_text2 = processor.apply_chat_template(messages2, add_generation_prompt=True)
inputs2 = processor(text=prompt_text2, images=[image], return_tensors="pt")

with torch.no_grad():
    outputs2 = model.generate(**inputs2, max_new_tokens=100, do_sample=False)

response2 = processor.decode(outputs2[0], skip_special_tokens=True)
print(f"\n=== TEST 2: JSON Output ===")
print(f"Response: {response2}")

print("\n5. Done!")
