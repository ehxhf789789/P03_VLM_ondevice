"""
다양한 로컬 VLM 모델 벤치마크
건설현장 자재 손상 평가 비교 테스트
"""
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'

import torch
from PIL import Image

# ============================================================================
# 모델 정의
# ============================================================================

@dataclass
class ModelConfig:
    name: str
    model_id: str
    processor_class: str  # 프로세서 타입
    model_class: str      # 모델 타입
    chat_format: str      # 채팅 템플릿 형식
    size_mb: int          # 대략적인 크기 (MB)
    notes: str

# 테스트할 로컬 VLM 모델 목록
LOCAL_VLM_MODELS = [
    ModelConfig(
        name="SmolVLM-256M",
        model_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        processor_class="AutoProcessor",
        model_class="AutoModelForImageTextToText",
        chat_format="smolvlm",
        size_mb=500,
        notes="HuggingFace 최소형 VLM"
    ),
    ModelConfig(
        name="SmolVLM-500M",
        model_id="HuggingFaceTB/SmolVLM-500M-Instruct",
        processor_class="AutoProcessor",
        model_class="AutoModelForImageTextToText",
        chat_format="smolvlm",
        size_mb=800,
        notes="HuggingFace 경량 VLM"
    ),
    ModelConfig(
        name="Moondream2",
        model_id="vikhyatk/moondream2",
        processor_class="AutoProcessor",
        model_class="AutoModelForCausalLM",
        chat_format="moondream",
        size_mb=2000,
        notes="Moondream 1.6B, 빠른 추론"
    ),
    ModelConfig(
        name="SmolVLM-2B",
        model_id="HuggingFaceTB/SmolVLM-Instruct",
        processor_class="AutoProcessor",
        model_class="AutoModelForImageTextToText",
        chat_format="smolvlm",
        size_mb=3500,
        notes="HuggingFace 2B VLM"
    ),
]

# ============================================================================
# 프롬프트 정의
# ============================================================================

PROMPTS = {
    "object_recognition_en": """What object is shown in this image?
Answer format:
- Object name:
- Type: (hardhat/safety boots/gloves/tool)
- Color:""",

    "damage_detection_en": """Analyze this construction equipment image for damage.

Check and report:
1. Object type
2. Damage presence (yes/no)
3. Damage type (crack, scratch, discoloration, wear, rust, etc.)
4. Damage location (top/center/bottom)
5. Severity (minor/moderate/severe)
6. Usability (usable/needs inspection/replace)""",

    "structured_json": """Analyze this image and output ONLY valid JSON:
{
    "object": {"name": "object name", "type": "hardhat|boots|gloves|tool", "color": "color"},
    "condition": {"overall": "new|good|fair|poor|damaged", "usable": true/false},
    "damages": [{"type": "damage type", "location": "where", "severity": "minor|moderate|severe"}]
}"""
}

# ============================================================================
# 모델별 추론 함수
# ============================================================================

def infer_smolvlm(model, processor, image, prompt, device):
    """SmolVLM 모델 추론"""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    full_response = processor.decode(outputs[0], skip_special_tokens=False)
    if "Assistant:" in full_response:
        response = full_response.split("Assistant:")[-1].strip()
        if "<end_of_utterance>" in response:
            response = response.split("<end_of_utterance>")[0].strip()
    else:
        response = full_response[-500:]

    return response

def infer_moondream(model, processor, image, prompt, device):
    """Moondream2 모델 추론"""
    # Moondream 특수 인터페이스
    try:
        # 새로운 버전 API
        encoded = model.encode_image(image)
        response = model.answer_question(encoded, prompt, processor)
    except:
        # 이전 버전 API
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300)
        response = processor.decode(outputs[0], skip_special_tokens=True)

    return response

def get_inference_function(chat_format: str) -> Callable:
    """채팅 형식에 따른 추론 함수 반환"""
    if chat_format == "smolvlm":
        return infer_smolvlm
    elif chat_format == "moondream":
        return infer_moondream
    else:
        return infer_smolvlm  # 기본값

# ============================================================================
# 테스트 함수
# ============================================================================

def load_test_images():
    """테스트 이미지 로드"""
    images = {}
    test_dir = Path("test_images")

    if not test_dir.exists():
        print("Error: test_images 폴더가 없습니다.")
        return images

    for img_path in test_dir.glob("*.jpg"):
        try:
            img = Image.open(img_path).convert("RGB")
            images[img_path.stem] = {"image": img, "path": str(img_path)}
        except Exception as e:
            print(f"  Failed: {img_path.name}")

    return images

def test_single_model(config: ModelConfig, images: Dict, device: str) -> Dict:
    """단일 모델 테스트"""
    from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM

    print(f"\n{'='*70}")
    print(f"Model: {config.name}")
    print(f"ID: {config.model_id}")
    print(f"Size: ~{config.size_mb}MB")
    print(f"{'='*70}")

    result = {
        "model": config.name,
        "model_id": config.model_id,
        "size_mb": config.size_mb,
        "device": device,
        "tests": [],
        "summary": {},
        "scores": {}
    }

    try:
        print("\n모델 로딩 중...")
        start_load = time.time()

        processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)

        if config.model_class == "AutoModelForCausalLM":
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                config.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )

        model = model.to(device)
        load_time = time.time() - start_load

        print(f"모델 로딩 완료: {load_time:.1f}초")
        result["summary"]["load_time_sec"] = round(load_time, 2)

    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        result["error"] = str(e)
        return result

    # 추론 함수 선택
    infer_fn = get_inference_function(config.chat_format)

    total_tests = 0
    successful_tests = 0
    total_time = 0

    # 점수 초기화
    scores = {"object_recognition": 0, "damage_detection": 0, "json_output": 0}

    for img_name, img_data in images.items():
        print(f"\n--- Image: {img_name} ---")
        image = img_data["image"]

        for prompt_name, prompt_text in PROMPTS.items():
            print(f"  [{prompt_name}]...", end=" ", flush=True)
            total_tests += 1

            try:
                start_time = time.time()
                response = infer_fn(model, processor, image, prompt_text, device)
                elapsed = time.time() - start_time
                total_time += elapsed

                test_result = {
                    "image": img_name,
                    "prompt": prompt_name,
                    "response": response[:500],
                    "time_sec": round(elapsed, 2),
                    "success": True
                }

                # 점수 계산
                response_lower = response.lower()
                if "object" in prompt_name or "recognition" in prompt_name:
                    keywords = ["hardhat", "helmet", "boot", "glove", "hammer", "tool", "safety"]
                    if any(kw in response_lower for kw in keywords):
                        scores["object_recognition"] += 1
                elif "damage" in prompt_name:
                    damage_words = ["damage", "crack", "scratch", "rust", "wear", "good", "new", "no damage"]
                    if any(word in response_lower for word in damage_words):
                        scores["damage_detection"] += 1
                elif "json" in prompt_name:
                    try:
                        if "{" in response and "}" in response:
                            start = response.find("{")
                            end = response.rfind("}") + 1
                            json.loads(response[start:end])
                            scores["json_output"] += 1
                    except:
                        pass

                successful_tests += 1
                print(f"OK ({elapsed:.1f}s)")
                print(f"      >> {response[:80]}...")

            except Exception as e:
                test_result = {
                    "image": img_name,
                    "prompt": prompt_name,
                    "error": str(e)[:200],
                    "success": False
                }
                print(f"FAIL: {str(e)[:50]}")

            result["tests"].append(test_result)

    # 요약
    result["summary"]["total_tests"] = total_tests
    result["summary"]["successful_tests"] = successful_tests
    result["summary"]["success_rate"] = round(successful_tests / total_tests * 100, 1) if total_tests > 0 else 0
    result["summary"]["avg_time_sec"] = round(total_time / successful_tests, 2) if successful_tests > 0 else 0
    result["summary"]["total_time_sec"] = round(total_time, 2)
    result["scores"] = scores

    # 메모리 정리
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result

# ============================================================================
# 시각화 함수
# ============================================================================

def visualize_all_results(all_results: List[Dict], output_prefix: str = "multi_vlm"):
    """모든 모델 결과 시각화"""
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Local VLM Model Comparison - Construction Equipment Damage Assessment",
                 fontsize=16, fontweight='bold')

    models = [r["model"] for r in all_results if "summary" in r]
    valid_results = [r for r in all_results if "summary" in r]

    if not models:
        print("No valid results to visualize")
        return

    # 색상 팔레트
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    # 1. 정확도 비교 (막대 차트)
    ax1 = fig.add_subplot(2, 2, 1)

    categories = ['Object Recognition', 'Damage Detection', 'JSON Output']
    x = np.arange(len(categories))
    width = 0.8 / len(models)

    for i, r in enumerate(valid_results):
        scores = [
            r["scores"]["object_recognition"],
            r["scores"]["damage_detection"],
            r["scores"]["json_output"]
        ]
        ax1.bar(x + i * width, scores, width, label=r["model"], color=colors[i])

    ax1.set_ylabel('Score (max 6)')
    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels(categories)
    ax1.set_title('Accuracy by Category')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 7)

    # 2. 응답 시간 비교
    ax2 = fig.add_subplot(2, 2, 2)

    avg_times = [r["summary"]["avg_time_sec"] for r in valid_results]
    load_times = [r["summary"]["load_time_sec"] for r in valid_results]

    x = np.arange(len(models))
    ax2.bar(x, load_times, label='Model Loading', color='lightgray', edgecolor='gray')
    ax2.bar(x, avg_times, bottom=load_times, label='Avg Inference', color='steelblue')

    ax2.set_ylabel('Time (seconds)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=20, ha='right')
    ax2.set_title('Time Performance')
    ax2.legend()

    for i, (t, l) in enumerate(zip(avg_times, load_times)):
        ax2.text(i, t + l + 0.5, f'{t+l:.1f}s', ha='center', fontsize=9)

    # 3. 모델 크기 vs 총점
    ax3 = fig.add_subplot(2, 2, 3)

    sizes = [r["size_mb"] for r in valid_results]
    total_scores = [sum(r["scores"].values()) for r in valid_results]

    scatter = ax3.scatter(sizes, total_scores, s=300, c=colors[:len(models)], alpha=0.7, edgecolors='black')

    for i, (model, size, score) in enumerate(zip(models, sizes, total_scores)):
        ax3.annotate(model, (size, score), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9)

    ax3.set_xlabel('Model Size (MB)')
    ax3.set_ylabel('Total Score (max 18)')
    ax3.set_title('Model Size vs Performance')
    ax3.grid(True, alpha=0.3)

    # 4. 성공률 비교
    ax4 = fig.add_subplot(2, 2, 4)

    success_rates = [r["summary"]["success_rate"] for r in valid_results]

    bars = ax4.bar(models, success_rates, color=colors, edgecolor='black')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('Inference Success Rate')
    ax4.set_ylim(0, 110)

    for bar, rate in zip(bars, success_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.0f}%', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_file = f"{output_prefix}_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved: {output_file}")

    # 상세 결과 테이블 생성
    create_result_table(valid_results, f"{output_prefix}_table.png")

    return output_file

def create_result_table(results: List[Dict], output_file: str):
    """결과 테이블 이미지 생성"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # 테이블 데이터 준비
    headers = ['Model', 'Size', 'Load(s)', 'Infer(s)', 'Object', 'Damage', 'JSON', 'Total', 'Success']
    rows = []

    for r in results:
        total = sum(r["scores"].values())
        rows.append([
            r["model"],
            f'{r["size_mb"]}MB',
            f'{r["summary"]["load_time_sec"]:.1f}',
            f'{r["summary"]["avg_time_sec"]:.1f}',
            f'{r["scores"]["object_recognition"]}/6',
            f'{r["scores"]["damage_detection"]}/6',
            f'{r["scores"]["json_output"]}/6',
            f'{total}/18',
            f'{r["summary"]["success_rate"]:.0f}%'
        ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['lightsteelblue'] * len(headers)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    plt.title("Local VLM Benchmark Results Summary", fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Table saved: {output_file}")

# ============================================================================
# 메인
# ============================================================================

def main(models_to_test: List[str] = None):
    """메인 실행"""
    print("=" * 70)
    print("Local VLM Multi-Model Benchmark")
    print("Construction Equipment Damage Assessment")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 이미지 로드
    print("\nLoading test images...")
    images = load_test_images()
    if not images:
        print("No test images found!")
        return

    print(f"Loaded: {len(images)} images")

    # 테스트할 모델 필터링
    if models_to_test:
        models = [m for m in LOCAL_VLM_MODELS if m.name in models_to_test]
    else:
        models = LOCAL_VLM_MODELS

    print(f"Models to test: {len(models)}")
    for m in models:
        print(f"  - {m.name} ({m.size_mb}MB)")

    # 벤치마크 실행
    all_results = []

    for config in models:
        result = test_single_model(config, images, device)
        all_results.append(result)

        # 중간 저장
        with open("multi_vlm_benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "device": device,
                "results": all_results
            }, f, indent=2, ensure_ascii=False)

    # 시각화
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    visualize_all_results(all_results)

    # 최종 요약
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    for r in all_results:
        if "summary" in r:
            total = sum(r["scores"].values())
            print(f"\n{r['model']} ({r['size_mb']}MB):")
            print(f"  Score: {total}/18 ({total/18*100:.1f}%)")
            print(f"  - Object: {r['scores']['object_recognition']}/6")
            print(f"  - Damage: {r['scores']['damage_detection']}/6")
            print(f"  - JSON: {r['scores']['json_output']}/6")
            print(f"  Time: {r['summary']['avg_time_sec']:.1f}s avg")
        elif "error" in r:
            print(f"\n{r['model']}: FAILED - {r['error'][:50]}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_results

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 특정 모델만 테스트
        models = sys.argv[1:]
        main(models)
    else:
        # SmolVLM 모델들만 먼저 테스트 (빠름)
        print("Testing SmolVLM models first (faster)...")
        print("To test specific models: python test_multiple_vlm.py SmolVLM-256M SmolVLM-500M")
        main(["SmolVLM-256M", "SmolVLM-500M"])
