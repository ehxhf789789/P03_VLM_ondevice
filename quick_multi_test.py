"""
빠른 다중 VLM 테스트 - 결과 저장 보장
"""
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import json
import time
import torch
from PIL import Image
from pathlib import Path
from datetime import datetime

def load_images():
    images = {}
    for p in Path("test_images").glob("*.jpg"):
        try:
            images[p.stem] = Image.open(p).convert("RGB")
        except:
            pass
    return images

def test_smolvlm(model_id, image, prompt):
    """SmolVLM 테스트"""
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    response = processor.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()

    del model, processor
    return response

def test_florence(model_id, image):
    """Florence-2 테스트"""
    from transformers import AutoProcessor, AutoModelForCausalLM

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    # Florence 태스크
    tasks = ["<CAPTION>", "<DETAILED_CAPTION>", "<OD>"]
    results = {}

    for task in tasks:
        inputs = processor(text=task, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)
        response = processor.decode(outputs[0], skip_special_tokens=False)
        # 결과 파싱
        parsed = processor.post_process_generation(response, task=task, image_size=(image.width, image.height))
        results[task] = str(parsed)

    del model, processor
    return results

PROMPT = """What is this object? Is there any damage?
Answer:
- Object: (hardhat/boots/gloves/hammer)
- Color:
- Damaged: (yes/no)
- Damage type: (if any)"""

def main():
    print("=" * 60)
    print("Quick Multi-VLM Test")
    print("=" * 60)

    images = load_images()
    print(f"Images: {len(images)}")

    # 테스트할 모델
    models = [
        ("SmolVLM-256M", "HuggingFaceTB/SmolVLM-256M-Instruct", "smolvlm"),
        ("SmolVLM-500M", "HuggingFaceTB/SmolVLM-500M-Instruct", "smolvlm"),
        ("Florence-2-base", "microsoft/Florence-2-base", "florence"),
    ]

    all_results = []

    for model_name, model_id, model_type in models:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")

        result = {
            "model": model_name,
            "model_id": model_id,
            "tests": [],
            "scores": {"object": 0, "damage": 0}
        }

        # 2개 이미지만 테스트 (빠른 테스트)
        test_images = list(images.items())[:2]

        for img_name, image in test_images:
            print(f"\n  Image: {img_name}...", end=" ", flush=True)

            try:
                start = time.time()

                if model_type == "smolvlm":
                    response = test_smolvlm(model_id, image, PROMPT)
                elif model_type == "florence":
                    response = str(test_florence(model_id, image))
                else:
                    response = "Unknown model type"

                elapsed = time.time() - start

                # 점수
                resp_lower = response.lower()
                if any(w in resp_lower for w in ["hardhat", "helmet", "boot", "glove", "hammer", "tool"]):
                    result["scores"]["object"] += 1
                if any(w in resp_lower for w in ["damage", "crack", "no damage", "good", "yes", "no"]):
                    result["scores"]["damage"] += 1

                result["tests"].append({
                    "image": img_name,
                    "response": response[:300],
                    "time": round(elapsed, 2),
                    "success": True
                })

                print(f"OK ({elapsed:.1f}s)")
                print(f"      >> {response[:100]}...")

            except Exception as e:
                result["tests"].append({
                    "image": img_name,
                    "error": str(e)[:100],
                    "success": False
                })
                print(f"FAIL: {str(e)[:50]}")

        all_results.append(result)

        # 중간 저장
        with open("quick_vlm_results.json", "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": all_results
            }, f, indent=2, ensure_ascii=False)

    # 최종 요약
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for r in all_results:
        total = r["scores"]["object"] + r["scores"]["damage"]
        max_score = len([t for t in r["tests"] if t.get("success", False)]) * 2
        print(f"\n{r['model']}:")
        print(f"  Object Recognition: {r['scores']['object']}")
        print(f"  Damage Detection: {r['scores']['damage']}")
        print(f"  Total: {total}/{max_score}")

    print(f"\nResults saved to: quick_vlm_results.json")

if __name__ == "__main__":
    main()
