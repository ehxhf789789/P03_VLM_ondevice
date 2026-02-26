"""SmolVLM 테스트 - 전체 출력"""
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

# 테스트 이미지 생성
img = np.zeros((224, 224, 3), dtype=np.uint8)
img[50:150, 50:150] = [255, 0, 0]  # 빨간 사각형
image = Image.fromarray(img)

# 테스트
prompt = "What is in this image? Describe the shapes and colors."
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)

print(f"\n--- Prompt template ---")
print(prompt_text[:500])
print("...")

inputs = processor(text=prompt_text, images=[image], return_tensors="pt")

print(f"\n--- Input shapes ---")
for k, v in inputs.items():
    if hasattr(v, 'shape'):
        print(f"{k}: {v.shape}")

print(f"\n--- Generating ---")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id
    )

print(f"Output shape: {outputs.shape}")
print(f"Output tokens: {outputs[0].tolist()[:20]}...")

# 전체 디코딩
full_response = processor.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Full response (with special tokens) ---")
print(full_response)

# 특수 토큰 제거
clean_response = processor.decode(outputs[0], skip_special_tokens=True)
print(f"\n--- Clean response ---")
print(clean_response)
