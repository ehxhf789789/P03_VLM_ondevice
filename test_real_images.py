"""
실제 건설현장 자재 이미지로 VLM 테스트
"""
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import json
import time
import torch
from PIL import Image
from pathlib import Path
from datetime import datetime

# 테스트할 실제 이미지
REAL_IMAGES = {
    "construction_site_hardhats": {
        "file": "hardhat_yellow.jpg",
        "description": "건설현장 작업자들 - 안전모 착용",
        "expected": "안전모, 안전조끼, 건설현장"
    },
    "construction_workers": {
        "file": "hardhat_single.jpg",
        "description": "건설현장 안전모 착용 작업자",
        "expected": "안전모, 안전조끼"
    },
    "hammer_claw": {
        "file": "hammer.jpg",
        "description": "클로 해머 (장도리)",
        "expected": "해머, 공구"
    },
    "wrench_set": {
        "file": "wrench.jpg",
        "description": "렌치/스패너 세트",
        "expected": "렌치, 스패너, 공구"
    },
    "worker_with_gloves": {
        "file": "gloves_work.jpg",
        "description": "작업장갑 착용 작업자",
        "expected": "장갑, 안전모"
    }
}

PROMPT = """Analyze this construction site image.

Identify:
1. What safety equipment or tools are visible? (hardhat, safety vest, boots, gloves, hammer, wrench, etc.)
2. What is the condition of the equipment? (new, used, worn, damaged)
3. Any visible damage or wear? Describe location and type.

Output JSON:
{
    "objects": ["list of identified items"],
    "conditions": {"item": "condition"},
    "damage": {"item": "damage description or none"}
}"""

def test_smolvlm(model_id, image, prompt):
    """SmolVLM 테스트"""
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)

    response = processor.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()

    del model, processor
    return response

def evaluate_response(response, expected):
    """응답 평가"""
    resp_lower = response.lower()

    scores = {
        "object_recognition": 0,
        "condition_assessment": 0,
        "json_format": 0,
        "detail_level": 0
    }

    # 객체 인식
    object_keywords = ["hardhat", "helmet", "안전모", "vest", "조끼", "boot", "신발",
                       "glove", "장갑", "hammer", "해머", "wrench", "렌치", "spanner",
                       "tool", "공구", "safety", "construction", "worker"]
    for kw in object_keywords:
        if kw in resp_lower:
            scores["object_recognition"] += 1
    scores["object_recognition"] = min(scores["object_recognition"], 5)

    # 상태 평가
    condition_words = ["new", "used", "worn", "damaged", "good", "fair", "poor",
                       "scratch", "rust", "crack", "clean", "dirty"]
    for word in condition_words:
        if word in resp_lower:
            scores["condition_assessment"] += 1
    scores["condition_assessment"] = min(scores["condition_assessment"], 3)

    # JSON 형식
    if "{" in response and "}" in response:
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            json.loads(response[start:end])
            scores["json_format"] = 3
        except:
            scores["json_format"] = 1

    # 상세도
    if len(response) > 200:
        scores["detail_level"] = 2
    elif len(response) > 100:
        scores["detail_level"] = 1

    return scores

def main():
    print("=" * 70)
    print("VLM 실제 이미지 테스트")
    print("건설현장 자재/장비 인식 평가")
    print("=" * 70)
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 이미지 로드
    image_dir = Path("real_images")
    images = {}

    print("\n이미지 로드 중...")
    for name, info in REAL_IMAGES.items():
        path = image_dir / info["file"]
        if path.exists():
            try:
                img = Image.open(path).convert("RGB")
                images[name] = {"image": img, "info": info}
                print(f"  OK: {name} - {info['description']}")
            except Exception as e:
                print(f"  FAIL: {name} - {e}")
        else:
            print(f"  SKIP: {name} - file not found")

    print(f"\n로드된 이미지: {len(images)}개")

    if not images:
        print("테스트할 이미지가 없습니다!")
        return

    # 테스트할 모델
    models = [
        ("SmolVLM-256M", "HuggingFaceTB/SmolVLM-256M-Instruct"),
        ("SmolVLM-500M", "HuggingFaceTB/SmolVLM-500M-Instruct"),
    ]

    all_results = []

    for model_name, model_id in models:
        print(f"\n{'='*70}")
        print(f"모델: {model_name}")
        print(f"{'='*70}")

        result = {
            "model": model_name,
            "model_id": model_id,
            "tests": [],
            "total_scores": {"object_recognition": 0, "condition_assessment": 0,
                            "json_format": 0, "detail_level": 0}
        }

        for img_name, img_data in images.items():
            print(f"\n  [{img_name}]")
            print(f"  설명: {img_data['info']['description']}")

            try:
                start = time.time()
                response = test_smolvlm(model_id, img_data["image"], PROMPT)
                elapsed = time.time() - start

                scores = evaluate_response(response, img_data["info"]["expected"])

                for k, v in scores.items():
                    result["total_scores"][k] += v

                test_result = {
                    "image": img_name,
                    "description": img_data["info"]["description"],
                    "response": response,
                    "scores": scores,
                    "time_sec": round(elapsed, 2)
                }
                result["tests"].append(test_result)

                total = sum(scores.values())
                print(f"  시간: {elapsed:.1f}s | 점수: {total}/13")
                print(f"  응답: {response[:150]}...")

            except Exception as e:
                print(f"  오류: {str(e)[:100]}")
                result["tests"].append({
                    "image": img_name,
                    "error": str(e)[:200]
                })

        # 평균 점수
        num_tests = len([t for t in result["tests"] if "scores" in t])
        if num_tests > 0:
            result["avg_scores"] = {k: round(v/num_tests, 2) for k, v in result["total_scores"].items()}

        all_results.append(result)

    # 결과 저장
    output = {
        "timestamp": datetime.now().isoformat(),
        "image_source": "real_images (Unsplash, Pexels)",
        "num_images": len(images),
        "results": all_results
    }

    with open("real_image_vlm_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 요약
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)

    for r in all_results:
        print(f"\n{r['model']}:")
        if "avg_scores" in r:
            print(f"  객체 인식: {r['avg_scores']['object_recognition']:.1f}/5")
            print(f"  상태 평가: {r['avg_scores']['condition_assessment']:.1f}/3")
            print(f"  JSON 형식: {r['avg_scores']['json_format']:.1f}/3")
            print(f"  상세도: {r['avg_scores']['detail_level']:.1f}/2")
            total = sum(r['avg_scores'].values())
            print(f"  총점: {total:.1f}/13 ({total/13*100:.0f}%)")

    print(f"\n결과 저장: real_image_vlm_results.json")

if __name__ == "__main__":
    main()
