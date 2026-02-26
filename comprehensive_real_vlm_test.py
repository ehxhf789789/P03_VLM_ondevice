"""
VLM 종합 성능 테스트 - 실제 건설현장 이미지 사용
==============================================
테스트 대상 모델:
1. SmolVLM-256M (초경량)
2. SmolVLM-500M (경량)
3. Qwen2-VL-2B (다국어 지원)
4. Gemini API (무료 티어 - API 키 필요)

테스트 이미지:
- real_images/ 폴더의 실제 건설현장 사진
"""

import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

import json
import time
import torch
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 테스트 이미지 정의
REAL_IMAGES = {
    "hardhat_yellow": {
        "file": "hardhat_yellow.jpg",
        "description": "Construction workers with yellow hardhats",
        "category": "safety_helmet",
        "expected_objects": ["hardhat", "helmet", "safety vest", "worker"]
    },
    "hardhat_single": {
        "file": "hardhat_single.jpg",
        "description": "Single worker at construction site",
        "category": "safety_helmet",
        "expected_objects": ["hardhat", "worker", "construction"]
    },
    "hammer": {
        "file": "hammer.jpg",
        "description": "Claw hammer on wooden surface",
        "category": "tool",
        "expected_objects": ["hammer", "claw hammer", "tool", "wood"]
    },
    "wrench": {
        "file": "wrench.jpg",
        "description": "Wrench/spanner set",
        "category": "tool",
        "expected_objects": ["wrench", "spanner", "tool"]
    },
    "gloves_work": {
        "file": "gloves_work.jpg",
        "description": "Worker with gloves and hardhat",
        "category": "safety_gear",
        "expected_objects": ["gloves", "hardhat", "worker"]
    },
    "safety_boots": {
        "file": "safety_boots.jpg",
        "description": "Safety work boots",
        "category": "safety_boots",
        "expected_objects": ["boots", "safety boots", "footwear"]
    },
    "work_gloves": {
        "file": "work_gloves.jpg",
        "description": "Work gloves",
        "category": "safety_gear",
        "expected_objects": ["gloves", "work gloves", "safety"]
    }
}

# 평가 프롬프트
PROMPT = """Analyze this construction site image for safety equipment and tools.

Identify:
1. What objects are visible? (hardhat, safety vest, boots, gloves, hammer, wrench, etc.)
2. What is the condition of each item? (new, used, worn, damaged)
3. Any visible damage or wear? Describe location and type if present.

Output your response as JSON:
{
    "objects": [{"name": "object name", "condition": "condition", "notes": "any observations"}],
    "overall_condition": "good/fair/poor",
    "damage_detected": true/false,
    "damage_details": "description if any"
}"""

def load_real_images(image_dir: str = "real_images") -> Dict:
    """실제 이미지 로드"""
    images = {}
    image_path = Path(image_dir)

    for name, info in REAL_IMAGES.items():
        path = image_path / info["file"]
        if path.exists():
            try:
                img = Image.open(path).convert("RGB")
                images[name] = {"image": img, "info": info}
                print(f"  OK: {name}")
            except Exception as e:
                print(f"  FAIL: {name} - {e}")
        else:
            # 파일이 없으면 비슷한 이름 찾기
            similar = list(image_path.glob(f"*{info['category']}*")) + list(image_path.glob("*.jpg"))
            if similar:
                try:
                    img = Image.open(similar[0]).convert("RGB")
                    images[name] = {"image": img, "info": info}
                    print(f"  OK: {name} (using {similar[0].name})")
                except:
                    print(f"  SKIP: {name} - file not found")
            else:
                print(f"  SKIP: {name} - file not found")

    return images

# ============================================================================
# Model Test Functions
# ============================================================================

def test_smolvlm(model_id: str, image: Image.Image, prompt: str) -> Dict:
    """SmolVLM 모델 테스트"""
    from transformers import AutoProcessor, AutoModelForImageTextToText

    result = {"success": False, "response": "", "time_sec": 0, "error": None}

    try:
        start = time.time()

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id)

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False)

        response = processor.decode(outputs[0], skip_special_tokens=True)
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        result["time_sec"] = round(time.time() - start, 2)
        result["response"] = response
        result["success"] = True

        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        result["error"] = str(e)[:200]

    return result

def test_qwen2vl(model_id: str, image: Image.Image, prompt: str) -> Dict:
    """Qwen2-VL 모델 테스트"""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    result = {"success": False, "response": "", "time_sec": 0, "error": None}

    try:
        start = time.time()

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        processor = AutoProcessor.from_pretrained(model_id)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=400)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        result["time_sec"] = round(time.time() - start, 2)
        result["response"] = response
        result["success"] = True

        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except ImportError:
        result["error"] = "qwen_vl_utils not installed. Run: pip install qwen-vl-utils"
    except Exception as e:
        result["error"] = str(e)[:200]

    return result

def test_gemini(image: Image.Image, prompt: str) -> Dict:
    """Google Gemini API 테스트"""
    result = {"success": False, "response": "", "time_sec": 0, "error": None}

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        result["error"] = "GEMINI_API_KEY not set"
        return result

    try:
        import google.generativeai as genai

        start = time.time()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        response = model.generate_content([prompt, image])

        result["time_sec"] = round(time.time() - start, 2)
        result["response"] = response.text
        result["success"] = True

    except ImportError:
        result["error"] = "google-generativeai not installed"
    except Exception as e:
        result["error"] = str(e)[:200]

    return result

# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_response(response: str, expected_objects: List[str]) -> Dict:
    """응답 품질 평가"""
    resp_lower = response.lower()

    scores = {
        "object_recognition": 0,
        "condition_assessment": 0,
        "json_format": 0,
        "detail_level": 0
    }

    # 1. Object Recognition (max 5)
    object_keywords = [
        "hardhat", "helmet", "safety", "vest", "boot", "shoe", "glove",
        "hammer", "wrench", "spanner", "tool", "construction", "worker",
        "protective", "equipment", "gear"
    ]
    for kw in object_keywords:
        if kw in resp_lower:
            scores["object_recognition"] += 1
    scores["object_recognition"] = min(scores["object_recognition"], 5)

    # Check expected objects
    for obj in expected_objects:
        if obj.lower() in resp_lower:
            scores["object_recognition"] = min(scores["object_recognition"] + 1, 5)

    # 2. Condition Assessment (max 3)
    condition_words = [
        "new", "used", "worn", "damaged", "good", "fair", "poor",
        "scratch", "rust", "crack", "clean", "dirty", "condition"
    ]
    for word in condition_words:
        if word in resp_lower:
            scores["condition_assessment"] += 1
    scores["condition_assessment"] = min(scores["condition_assessment"], 3)

    # 3. JSON Format (max 3)
    if "{" in response and "}" in response:
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            json.loads(response[start:end])
            scores["json_format"] = 3  # Valid JSON
        except:
            scores["json_format"] = 1  # Attempted JSON

    # 4. Detail Level (max 2)
    if len(response) > 300:
        scores["detail_level"] = 2
    elif len(response) > 150:
        scores["detail_level"] = 1

    return scores

# ============================================================================
# Main Benchmark
# ============================================================================

def run_comprehensive_test():
    """종합 VLM 테스트 실행"""
    print("=" * 70)
    print("VLM Comprehensive Real Image Benchmark")
    print("Construction Equipment Recognition & Damage Assessment")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load images
    print("\nLoading real images...")
    images = load_real_images()
    print(f"Loaded: {len(images)} images")

    if not images:
        print("No images found! Exiting.")
        return

    # Models to test
    models = [
        ("SmolVLM-256M", "HuggingFaceTB/SmolVLM-256M-Instruct", "smolvlm"),
        ("SmolVLM-500M", "HuggingFaceTB/SmolVLM-500M-Instruct", "smolvlm"),
        ("Gemini-2.0-Flash", "gemini-2.0-flash", "gemini"),
    ]

    # Check if Qwen2-VL can be tested
    try:
        from qwen_vl_utils import process_vision_info
        models.append(("Qwen2-VL-2B", "Qwen/Qwen2-VL-2B-Instruct", "qwen"))
        print("\nQwen2-VL-2B available for testing")
    except ImportError:
        print("\nQwen2-VL-2B skipped (qwen_vl_utils not installed)")
        print("  To enable: pip install qwen-vl-utils transformers>=4.37.0")

    all_results = []

    # Test each model
    for model_name, model_id, model_type in models:
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print(f"{'=' * 70}")

        # Skip Gemini if no API key
        if model_type == "gemini":
            if not os.environ.get("GEMINI_API_KEY"):
                print("  SKIPPED: GEMINI_API_KEY not set")
                print("  To test Gemini (free):")
                print("    1. Get key: https://aistudio.google.com/apikey")
                print("    2. Set: set GEMINI_API_KEY=your_key")
                continue

        result = {
            "model": model_name,
            "model_id": model_id,
            "model_type": model_type,
            "tests": [],
            "total_scores": {"object_recognition": 0, "condition_assessment": 0,
                           "json_format": 0, "detail_level": 0}
        }

        # Test on subset of images (max 4 for speed)
        test_images = list(images.items())[:4]

        for img_name, img_data in test_images:
            print(f"\n  [{img_name}]")
            print(f"  Description: {img_data['info']['description']}")

            try:
                if model_type == "smolvlm":
                    test_result = test_smolvlm(model_id, img_data["image"], PROMPT)
                elif model_type == "qwen":
                    test_result = test_qwen2vl(model_id, img_data["image"], PROMPT)
                elif model_type == "gemini":
                    test_result = test_gemini(img_data["image"], PROMPT)
                else:
                    test_result = {"success": False, "error": "Unknown model type"}

                if test_result["success"]:
                    scores = evaluate_response(
                        test_result["response"],
                        img_data["info"]["expected_objects"]
                    )

                    for k, v in scores.items():
                        result["total_scores"][k] += v

                    total = sum(scores.values())
                    print(f"  Time: {test_result['time_sec']}s | Score: {total}/13")
                    print(f"  Response: {test_result['response'][:150]}...")

                    result["tests"].append({
                        "image": img_name,
                        "description": img_data["info"]["description"],
                        "response": test_result["response"],
                        "scores": scores,
                        "time_sec": test_result["time_sec"]
                    })
                else:
                    print(f"  ERROR: {test_result['error']}")
                    result["tests"].append({
                        "image": img_name,
                        "error": test_result["error"]
                    })

            except Exception as e:
                print(f"  ERROR: {str(e)[:100]}")
                result["tests"].append({
                    "image": img_name,
                    "error": str(e)[:200]
                })

        # Calculate averages
        num_successful = len([t for t in result["tests"] if "scores" in t])
        if num_successful > 0:
            result["avg_scores"] = {k: round(v / num_successful, 2)
                                   for k, v in result["total_scores"].items()}
            result["avg_total"] = round(sum(result["avg_scores"].values()), 2)

        all_results.append(result)

        # Save intermediate results
        save_results(all_results)

    # Final summary
    print_summary(all_results)

    return all_results

def save_results(results: List[Dict]):
    """결과 저장"""
    output = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_type": "comprehensive_real_images",
        "image_source": "real_images/ folder (Unsplash, Pexels)",
        "results": results
    }

    with open("comprehensive_vlm_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

def print_summary(results: List[Dict]):
    """결과 요약 출력"""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    max_total = 13  # 5 + 3 + 3 + 2

    for r in results:
        print(f"\n{r['model']}:")
        if "avg_scores" in r:
            print(f"  Object Recognition: {r['avg_scores']['object_recognition']:.1f}/5")
            print(f"  Condition Assessment: {r['avg_scores']['condition_assessment']:.1f}/3")
            print(f"  JSON Format: {r['avg_scores']['json_format']:.1f}/3")
            print(f"  Detail Level: {r['avg_scores']['detail_level']:.1f}/2")
            pct = r['avg_total'] / max_total * 100
            print(f"  TOTAL: {r['avg_total']:.1f}/{max_total} ({pct:.0f}%)")
        else:
            successful = len([t for t in r["tests"] if "scores" in t])
            print(f"  Tests: {successful}/{len(r['tests'])} successful")

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Rank by score
    ranked = sorted(
        [(r["model"], r.get("avg_total", 0)) for r in results],
        key=lambda x: x[1],
        reverse=True
    )

    print("\nModel Ranking (by total score):")
    for i, (model, score) in enumerate(ranked, 1):
        print(f"  {i}. {model}: {score:.1f}/13")

    print("\nResults saved to: comprehensive_vlm_results.json")

if __name__ == "__main__":
    run_comprehensive_test()
