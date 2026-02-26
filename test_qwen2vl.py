"""
Qwen2-VL-2B 테스트 - 실제 건설현장 이미지
"""
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import json
import time
import torch
from PIL import Image
from pathlib import Path
from datetime import datetime

PROMPT = """Analyze this construction site image.

Identify:
1. What objects are visible? (hardhat, safety vest, boots, gloves, hammer, wrench, etc.)
2. What is the condition of each item? (new, used, worn, damaged)
3. Any visible damage or wear?

Output JSON:
{
    "objects": [{"name": "item", "condition": "condition"}],
    "damage_detected": true/false,
    "notes": "observations"
}"""

def test_qwen2vl_simple(model_id: str, image: Image.Image, prompt: str):
    """Qwen2-VL 간단 테스트"""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print(f"  Loading model: {model_id}")
    start = time.time()

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"  Model loaded: {time.time() - start:.1f}s")

    # Simple message format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Process
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # For Qwen2-VL, we need to handle images differently
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    gen_start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=300)

    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True
    )[0]

    total_time = time.time() - start
    gen_time = time.time() - gen_start

    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Total time: {total_time:.1f}s")

    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response, total_time

def main():
    print("=" * 60)
    print("Qwen2-VL-2B Test with Real Images")
    print("=" * 60)

    # Load images
    image_dir = Path("real_images")
    test_images = list(image_dir.glob("*.jpg"))[:3]  # Test first 3 images

    print(f"\nImages: {len(test_images)}")

    if not test_images:
        print("No images found!")
        return

    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    results = []

    for img_path in test_images:
        print(f"\n{'='*60}")
        print(f"Image: {img_path.name}")
        print(f"{'='*60}")

        try:
            image = Image.open(img_path).convert("RGB")
            response, elapsed = test_qwen2vl_simple(model_id, image, PROMPT)

            print(f"\nResponse:\n{response[:500]}")

            results.append({
                "image": img_path.name,
                "response": response,
                "time_sec": round(elapsed, 2),
                "success": True
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "image": img_path.name,
                "error": str(e)[:300],
                "success": False
            })

    # Save results
    output = {
        "model": "Qwen2-VL-2B",
        "timestamp": datetime.now().isoformat(),
        "tests": results
    }

    with open("qwen2vl_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n\nResults saved to: qwen2vl_results.json")

if __name__ == "__main__":
    main()
