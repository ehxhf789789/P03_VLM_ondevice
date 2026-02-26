"""실제 이미지로 SmolVLM 테스트"""
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

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    full_response = processor.decode(outputs[0], skip_special_tokens=False)
    if "Assistant:" in full_response:
        response = full_response.split("Assistant:")[-1].strip()
        if "<end_of_utterance>" in response:
            response = response.split("<end_of_utterance>")[0].strip()
    else:
        response = full_response

    print(f"RESPONSE:\n{response}")
    return response

# 실제 이미지들 테스트
test_images = [
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg", "cat"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Red_apple.jpg/800px-Red_apple.jpg", "apple"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/2008-07-25_White_cup_of_coffee_with_coffee_beans.jpg/800px-2008-07-25_White_cup_of_coffee_with_coffee_beans.jpg", "coffee"),
]

for url, name in test_images:
    print(f"\n\n{'#'*70}")
    print(f"TESTING IMAGE: {name}")
    print(f"{'#'*70}")

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            image_data = response.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        print(f"Image loaded: {image.size}")

        # 테스트 1: 간단한 설명
        run_test(image, "What is in this image?", "Simple Description")

        # 테스트 2: 상세 설명
        run_test(image, "Describe this image in detail. What do you see?", "Detailed Description")

        # 테스트 3: 상태 평가
        run_test(image,
            "Evaluate the condition of the object in this image. Is it damaged or in good condition? Describe any visible issues.",
            "Condition Assessment"
        )

    except Exception as e:
        print(f"Error loading {name}: {e}")

print("\n\n" + "="*70)
print("SUMMARY: SmolVLM-256M Capabilities")
print("="*70)
print("""
Based on testing, SmolVLM-256M-Instruct can:
✓ Basic object recognition (simple objects)
✓ Simple image descriptions
✓ Color and shape detection

Limitations:
✗ Complex damage analysis
✗ Precise location/bounding box coordinates
✗ Detailed product classification
✗ Structured JSON output (often copies template)
✗ Quantitative assessments (percentages, scores)

Recommendation:
- For production damage assessment, consider larger models:
  - SmolVLM-500M-Instruct
  - Qwen-VL
  - LLaVA 1.5/1.6
  - GPT-4V / Claude Vision (API)
""")
