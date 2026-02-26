"""
SmolVLM 로컬 모델 벤치마크
건설현장 자재 손상 평가 테스트
"""
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# 경고 메시지 최소화
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'  # 진행률은 표시

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# ============================================================================
# 설정
# ============================================================================

MODELS_TO_TEST = [
    ("SmolVLM-256M", "HuggingFaceTB/SmolVLM-256M-Instruct"),
    ("SmolVLM-500M", "HuggingFaceTB/SmolVLM-500M-Instruct"),
]

PROMPTS = {
    "object_recognition": """이 이미지에서 보이는 물체를 한국어로 식별하세요.
답변 형식:
- 물체 이름:
- 유형: (안전모/안전화/장갑/공구 중 선택)
- 색상:""",

    "damage_detection": """이 건설현장 자재 이미지를 분석하세요.

확인 항목:
1. 물체 종류
2. 손상 유무 (있음/없음)
3. 손상 유형 (균열, 스크래치, 변색, 마모, 녹 등)
4. 손상 위치 (상단/중앙/하단)
5. 손상 심각도 (경미/보통/심각)
6. 사용 가능 여부 (가능/점검필요/교체필요)""",

    "structured_json": """Analyze this construction equipment image and output ONLY valid JSON:
{
    "object": {"name": "object name", "type": "hardhat|boots|gloves|tool"},
    "condition": {"overall": "new|good|fair|poor|damaged", "usable": true/false},
    "damages": [{"type": "damage type", "severity": "minor|moderate|severe"}]
}"""
}

# ============================================================================
# 테스트 함수
# ============================================================================

def load_test_images():
    """test_images 폴더에서 이미지 로드"""
    images = {}
    test_dir = Path("test_images")

    if not test_dir.exists():
        print("Error: test_images 폴더가 없습니다. test_image_download.py를 먼저 실행하세요.")
        return images

    for img_path in test_dir.glob("*.jpg"):
        try:
            img = Image.open(img_path).convert("RGB")
            images[img_path.stem] = {"image": img, "path": str(img_path)}
            print(f"  Loaded: {img_path.name} ({img.size[0]}x{img.size[1]})")
        except Exception as e:
            print(f"  Failed: {img_path.name} - {e}")

    return images

def test_model(model_name, model_id, images, device):
    """단일 모델 테스트"""
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"ID: {model_id}")
    print(f"Device: {device}")
    print(f"{'='*70}")

    results = {
        "model": model_name,
        "model_id": model_id,
        "device": device,
        "tests": [],
        "summary": {}
    }

    try:
        print("\n모델 로딩 중...")
        start_load = time.time()

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        model = model.to(device)

        load_time = time.time() - start_load
        print(f"모델 로딩 완료: {load_time:.1f}초")
        results["summary"]["load_time_sec"] = round(load_time, 2)

    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        results["error"] = str(e)
        return results

    total_tests = 0
    successful_tests = 0
    total_time = 0

    for img_name, img_data in images.items():
        print(f"\n--- Image: {img_name} ---")
        image = img_data["image"]

        for prompt_name, prompt_text in PROMPTS.items():
            print(f"  [{prompt_name}]...", end=" ", flush=True)
            total_tests += 1

            try:
                start_time = time.time()

                # 채팅 형식으로 프롬프트 구성
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text}
                    ]}
                ]

                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=300,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id
                    )

                elapsed = time.time() - start_time
                total_time += elapsed

                # 응답 추출
                full_response = processor.decode(outputs[0], skip_special_tokens=False)
                if "Assistant:" in full_response:
                    response = full_response.split("Assistant:")[-1].strip()
                    if "<end_of_utterance>" in response:
                        response = response.split("<end_of_utterance>")[0].strip()
                else:
                    # 입력 부분 제거
                    input_len = len(processor.decode(inputs.input_ids[0]))
                    response = full_response[input_len:].strip()

                test_result = {
                    "image": img_name,
                    "prompt": prompt_name,
                    "response": response[:500],
                    "time_sec": round(elapsed, 2),
                    "success": True
                }

                successful_tests += 1
                print(f"OK ({elapsed:.1f}s)")
                print(f"      Response: {response[:100]}...")

            except Exception as e:
                test_result = {
                    "image": img_name,
                    "prompt": prompt_name,
                    "error": str(e),
                    "success": False
                }
                print(f"FAIL: {str(e)[:50]}")

            results["tests"].append(test_result)

    # 요약
    results["summary"]["total_tests"] = total_tests
    results["summary"]["successful_tests"] = successful_tests
    results["summary"]["success_rate"] = round(successful_tests / total_tests * 100, 1) if total_tests > 0 else 0
    results["summary"]["avg_time_sec"] = round(total_time / successful_tests, 2) if successful_tests > 0 else 0
    results["summary"]["total_time_sec"] = round(total_time, 2)

    # 메모리 정리
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results

# ============================================================================
# 평가 함수
# ============================================================================

def evaluate_results(results):
    """결과 평가 및 점수화"""
    scores = {
        "object_recognition": 0,
        "damage_detection": 0,
        "json_output": 0
    }

    for test in results.get("tests", []):
        if not test.get("success"):
            continue

        response = test.get("response", "").lower()
        prompt_type = test.get("prompt", "")

        if prompt_type == "object_recognition":
            # 객체 인식 키워드 확인
            keywords = ["안전모", "헬멧", "hardhat", "helmet", "부츠", "boots", "신발",
                       "장갑", "gloves", "해머", "hammer", "공구", "tool"]
            if any(kw in response for kw in keywords):
                scores["object_recognition"] += 1

        elif prompt_type == "damage_detection":
            # 손상 관련 키워드 확인
            damage_words = ["손상", "damage", "균열", "crack", "스크래치", "녹", "rust",
                           "양호", "good", "정상", "새것", "new"]
            if any(word in response for word in damage_words):
                scores["damage_detection"] += 1

        elif prompt_type == "structured_json":
            # JSON 파싱 가능 여부
            try:
                if "{" in response and "}" in response:
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    json.loads(response[start:end])
                    scores["json_output"] += 1
            except:
                pass

    return scores

# ============================================================================
# 메인
# ============================================================================

def main():
    print("=" * 70)
    print("SmolVLM 건설현장 자재 손상 평가 벤치마크")
    print("=" * 70)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 디바이스 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n사용 디바이스: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 이미지 로드
    print("\n테스트 이미지 로딩...")
    images = load_test_images()
    if not images:
        print("테스트 이미지가 없습니다!")
        return

    print(f"\n로드된 이미지: {len(images)}개")
    print(f"테스트할 모델: {len(MODELS_TO_TEST)}개")
    print(f"프롬프트 유형: {len(PROMPTS)}개")
    print(f"총 테스트 수: {len(images) * len(MODELS_TO_TEST) * len(PROMPTS)}개")

    # 벤치마크 실행
    all_results = []

    for model_name, model_id in MODELS_TO_TEST:
        result = test_model(model_name, model_id, images, device)
        scores = evaluate_results(result)
        result["scores"] = scores
        all_results.append(result)

    # 결과 저장
    output = {
        "benchmark_info": {
            "timestamp": datetime.now().isoformat(),
            "device": device,
            "num_images": len(images),
            "num_models": len(all_results),
            "prompts": list(PROMPTS.keys())
        },
        "images": list(images.keys()),
        "results": all_results
    }

    output_file = "smolvlm_benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 요약 출력
    print("\n" + "=" * 70)
    print("벤치마크 결과 요약")
    print("=" * 70)

    for result in all_results:
        print(f"\n{result['model']}:")
        if "summary" in result:
            s = result["summary"]
            print(f"  성공률: {s.get('success_rate', 0)}% ({s.get('successful_tests', 0)}/{s.get('total_tests', 0)})")
            print(f"  평균 응답 시간: {s.get('avg_time_sec', 0)}초")
            print(f"  모델 로딩 시간: {s.get('load_time_sec', 0)}초")

        if "scores" in result:
            scores = result["scores"]
            print(f"  평가 점수:")
            print(f"    - 객체 인식: {scores.get('object_recognition', 0)}")
            print(f"    - 손상 탐지: {scores.get('damage_detection', 0)}")
            print(f"    - JSON 출력: {scores.get('json_output', 0)}")

    print(f"\n결과 저장: {output_file}")
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_results

if __name__ == "__main__":
    main()
