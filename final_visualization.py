"""
VLM 벤치마크 최종 종합 시각화
모든 테스트 결과 통합
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_all_results():
    """모든 결과 파일 통합"""
    all_data = {}

    # 1. multi_vlm 결과
    if os.path.exists("multi_vlm_benchmark_results.json"):
        with open("multi_vlm_benchmark_results.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            for r in data.get("results", []):
                if "summary" in r and r["summary"]:
                    all_data[r["model"]] = {
                        "object": r["scores"]["object_recognition"],
                        "damage": r["scores"]["damage_detection"],
                        "json": r["scores"]["json_output"],
                        "time": r["summary"]["avg_time_sec"],
                        "size": r.get("size_mb", 500)
                    }

    # 2. quick 결과
    if os.path.exists("quick_vlm_results.json"):
        with open("quick_vlm_results.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            for r in data.get("results", []):
                if r["model"] not in all_data:
                    success_count = len([t for t in r["tests"] if t.get("success")])
                    if success_count > 0:
                        all_data[r["model"]] = {
                            "object": r["scores"]["object"],
                            "damage": r["scores"]["damage"],
                            "json": 0,
                            "time": sum(t.get("time", 0) for t in r["tests"] if t.get("success")) / success_count,
                            "size": 500 if "256" in r["model"] else 800
                        }

    return all_data

def create_final_chart():
    """최종 종합 차트 생성"""
    data = load_all_results()

    if not data:
        print("No results found!")
        return

    models = list(data.keys())
    print(f"Models found: {models}")

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("VLM 로컬 모델 Zero-shot 성능 비교\n건설현장 자재 손상 평가 (파인튜닝 없음)", fontsize=18, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    # 1. 종합 점수 비교
    ax1 = fig.add_subplot(2, 2, 1)

    categories = ['객체 인식', '손상 탐지', 'JSON 출력']
    x = np.arange(len(categories))
    width = 0.8 / max(len(models), 1)

    for i, (model, scores) in enumerate(data.items()):
        values = [scores["object"], scores["damage"], scores["json"]]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax1.bar(x + offset, values, width, label=model, color=colors[i])

    ax1.set_ylabel('점수')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_title('평가 항목별 점수', fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, 8)

    # 2. 총점 막대
    ax2 = fig.add_subplot(2, 2, 2)

    totals = {m: d["object"] + d["damage"] + d["json"] for m, d in data.items()}
    max_possible = 18  # 6 images * 3 categories

    bars = ax2.barh(list(totals.keys()), list(totals.values()), color=colors[:len(models)])
    ax2.set_xlabel(f'총점 (최대 {max_possible})')
    ax2.set_title('모델별 총점', fontweight='bold')

    for bar, score in zip(bars, totals.values()):
        pct = score / max_possible * 100
        ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{score} ({pct:.0f}%)', va='center', fontweight='bold')

    # 3. 응답 시간
    ax3 = fig.add_subplot(2, 2, 3)

    times = {m: d["time"] for m, d in data.items()}
    bars = ax3.bar(list(times.keys()), list(times.values()), color=colors[:len(models)])
    ax3.set_ylabel('평균 응답 시간 (초)')
    ax3.set_title('추론 속도', fontweight='bold')

    for bar, t in zip(bars, times.values()):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{t:.1f}s', ha='center', fontweight='bold')

    # 4. 결론 텍스트
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    conclusion = """
┌────────────────────────────────────────────────────────────────┐
│                    VLM Zero-shot 성능 평가 결론                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  테스트 모델                                                   │
│  ────────────────────────────────────────────────────────────  │
│  • SmolVLM-256M (256M): 객체 인식 실패, 손상 여부만 응답      │
│  • SmolVLM-500M (500M): 부분적 인식, 형식 이해 제한적         │
│  • Florence-2: API 호환성 문제로 테스트 실패                  │
│  • Moondream2: API 변경으로 테스트 실패                       │
│                                                                │
│  주요 발견                                                     │
│  ────────────────────────────────────────────────────────────  │
│  • 256M/500M 소형 모델은 건설 자재 인식에 부적합              │
│  • Zero-shot으로는 정확한 손상 위치/정도 파악 불가            │
│  • JSON 정형 출력 능력 매우 제한적                            │
│  • 합성 이미지 인식 정확도 낮음 (추상적 형태 이해 부족)       │
│                                                                │
│  권장 사항                                                     │
│  ────────────────────────────────────────────────────────────  │
│  1. Gemini API (무료) 사용하여 실제 성능 확인 권장            │
│  2. 7B+ 모델 (Qwen2.5-VL-7B, LLaVA-1.6-7B) 테스트 필요        │
│  3. 실제 건설 자재 이미지로 재테스트 필요                     │
│  4. 온디바이스 배포시 Fine-tuning 필수                        │
│                                                                │
│  다음 단계                                                     │
│  ────────────────────────────────────────────────────────────  │
│  • Gemini 2.0 Flash API 테스트 (무료, 고성능)                 │
│  • 실제 안전모/안전화 사진으로 테스트                         │
│  • Qwen2-VL-2B 또는 7B 모델 추가 테스트                       │
└────────────────────────────────────────────────────────────────┘
    """

    ax4.text(0.5, 0.5, conclusion, transform=ax4.transAxes,
             fontsize=10, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    output = "vlm_final_comparison.png"
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output}")

    # 결과 테이블도 출력
    print("\n" + "=" * 60)
    print("VLM 모델 성능 요약")
    print("=" * 60)
    print(f"{'모델':<20} {'객체':>8} {'손상':>8} {'JSON':>8} {'총점':>8} {'시간':>8}")
    print("-" * 60)

    for model, scores in data.items():
        total = scores["object"] + scores["damage"] + scores["json"]
        print(f"{model:<20} {scores['object']:>8} {scores['damage']:>8} {scores['json']:>8} {total:>8} {scores['time']:>7.1f}s")

    return output

if __name__ == "__main__":
    create_final_chart()
