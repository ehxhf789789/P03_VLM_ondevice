"""
VLM 모델 Zero-shot 성능 벤치마크
- 손상 위치 탐지
- 손상 정보 파악
- 객체 분류/인식
- 정형 데이터 출력
"""
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import numpy as np
import json
import time

# 테스트할 모델 목록
MODELS = [
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    # "Qwen/Qwen2-VL-2B-Instruct",  # 더 큰 모델
]

# 테스트 프롬프트
PROMPTS = {
    "object_recognition": "What object is shown in this image? Answer in one sentence.",

    "damage_detection": """Analyze this image for any damage or defects.
Describe: type of damage, severity (minor/moderate/severe), and location.""",

    "json_output": """Analyze this image and output ONLY valid JSON:
{
  "object": "identified object name",
  "condition": "new/good/fair/poor/damaged",
  "damages": [
    {"type": "damage type", "severity": "minor/moderate/severe", "location": "where"}
  ]
}""",

    "location_description": """Describe the location of any damage or notable features.
Use directional terms: top-left, top-center, top-right, center-left, center, center-right, bottom-left, bottom-center, bottom-right.
Also estimate the size as percentage of the image."""
}

def create_test_images():
    """테스트 이미지 생성"""
    images = {}

    # 1. 깨끗한 표면 (손상 없음)
    clean = np.ones((384, 384, 3), dtype=np.uint8) * 200
    images["clean_surface"] = Image.fromarray(clean)

    # 2. 스크래치가 있는 표면
    scratched = np.ones((384, 384, 3), dtype=np.uint8) * 180
    for i in range(50, 300):
        scratched[i, i:i+2] = [60, 50, 40]  # 대각선 스크래치
        scratched[i, 150:153] = [70, 60, 50]  # 세로 스크래치
    images["scratched_surface"] = Image.fromarray(scratched)

    # 3. 녹이 있는 표면
    rusted = np.ones((384, 384, 3), dtype=np.uint8) * 160
    rusted[80:180, 200:320] = [139, 90, 43]  # 녹 패치 (우측 상단)
    rusted[250:350, 50:150] = [120, 70, 30]  # 녹 패치 (좌측 하단)
    images["rusted_surface"] = Image.fromarray(rusted)

    # 4. 찌그러진/덴트
    dented = np.ones((384, 384, 3), dtype=np.uint8) * 190
    # 원형 덴트 시뮬레이션 (그라데이션)
    for y in range(150, 250):
        for x in range(150, 250):
            dist = np.sqrt((x-200)**2 + (y-200)**2)
            if dist < 50:
                shade = int(190 - (50 - dist) * 2)
                dented[y, x] = [shade, shade, shade]
    images["dented_surface"] = Image.fromarray(dented)

    # 5. 복합 손상
    complex_damage = np.ones((384, 384, 3), dtype=np.uint8) * 170
    # 스크래치
    for i in range(30, 150):
        complex_damage[i, i+50:i+53] = [50, 40, 30]
    # 녹
    complex_damage[50:120, 250:340] = [145, 85, 35]
    # 변색
    complex_damage[220:300, 100:200] = [150, 160, 140]
    images["complex_damage"] = Image.fromarray(complex_damage)

    return images

def test_model(model_id, images, device="cpu"):
    """단일 모델 테스트"""
    print(f"\n{'='*70}")
    print(f"Testing: {model_id}")
    print(f"{'='*70}")

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id)
        model = model.to(device)
        print(f"Model loaded on {device}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    results = {"model": model_id, "tests": []}

    for img_name, image in images.items():
        print(f"\n--- Image: {img_name} ---")

        for prompt_name, prompt in PROMPTS.items():
            start_time = time.time()

            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id
                )

            elapsed = time.time() - start_time

            full_response = processor.decode(outputs[0], skip_special_tokens=False)
            if "Assistant:" in full_response:
                response = full_response.split("Assistant:")[-1].strip()
                if "<end_of_utterance>" in response:
                    response = response.split("<end_of_utterance>")[0].strip()
            else:
                response = full_response[-500:]  # 마지막 500자

            result = {
                "image": img_name,
                "prompt": prompt_name,
                "response": response,
                "time_sec": round(elapsed, 2)
            }
            results["tests"].append(result)

            print(f"  [{prompt_name}] ({elapsed:.1f}s)")
            print(f"    Response: {response[:100]}...")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results

def evaluate_json_quality(response):
    """JSON 출력 품질 평가"""
    try:
        # JSON 추출 시도
        if "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]
            parsed = json.loads(json_str)

            # 필수 필드 확인
            has_object = "object" in parsed
            has_condition = "condition" in parsed
            has_damages = "damages" in parsed and isinstance(parsed.get("damages"), list)

            return {
                "valid_json": True,
                "has_object": has_object,
                "has_condition": has_condition,
                "has_damages": has_damages,
                "parsed": parsed
            }
    except:
        pass

    return {"valid_json": False}

def main():
    print("="*70)
    print("VLM Zero-shot Performance Benchmark")
    print("="*70)

    # 테스트 이미지 생성
    print("\nCreating test images...")
    images = create_test_images()
    print(f"Created {len(images)} test images")

    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    all_results = []

    for model_id in MODELS:
        result = test_model(model_id, images, device)
        if result:
            all_results.append(result)

    # 결과 요약
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    for result in all_results:
        print(f"\n{result['model']}:")

        # JSON 출력 품질
        json_tests = [t for t in result['tests'] if t['prompt'] == 'json_output']
        valid_json_count = sum(1 for t in json_tests if evaluate_json_quality(t['response'])['valid_json'])
        print(f"  JSON Output Quality: {valid_json_count}/{len(json_tests)} valid")

        # 평균 응답 시간
        avg_time = np.mean([t['time_sec'] for t in result['tests']])
        print(f"  Average Response Time: {avg_time:.2f}s")

    # 결과 저장
    with open("vlm_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\nResults saved to vlm_benchmark_results.json")

if __name__ == "__main__":
    main()
