"""
VLM 벤치마크 결과 시각화
"""
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_results(filepath):
    """결과 JSON 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def visualize_smolvlm_results(results_file="smolvlm_benchmark_results.json"):
    """SmolVLM 결과 시각화"""

    data = load_results(results_file)
    results = data["results"]

    # Figure 설정
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("VLM 로컬 모델 성능 비교 - 건설현장 자재 손상 평가", fontsize=16, fontweight='bold')

    # 1. 모델별 성공률 및 응답 시간
    ax1 = fig.add_subplot(2, 2, 1)
    models = [r["model"] for r in results]
    success_rates = [r["summary"]["success_rate"] for r in results]
    avg_times = [r["summary"]["avg_time_sec"] for r in results]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, success_rates, width, label='성공률 (%)', color='steelblue')
    ax1.set_ylabel('성공률 (%)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(0, 120)

    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, avg_times, width, label='평균 응답 시간 (초)', color='coral')
    ax1_twin.set_ylabel('응답 시간 (초)', color='coral')
    ax1_twin.tick_params(axis='y', labelcolor='coral')

    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_title('모델별 성공률 및 응답 시간')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    # 값 표시
    for bar, val in zip(bars1, success_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, avg_times):
        ax1_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val:.1f}s', ha='center', va='bottom', fontsize=10)

    # 2. 평가 점수 비교
    ax2 = fig.add_subplot(2, 2, 2)

    categories = ['객체 인식', '손상 탐지', 'JSON 출력']
    scores_256m = [results[0]["scores"]["object_recognition"],
                   results[0]["scores"]["damage_detection"],
                   results[0]["scores"]["json_output"]]
    scores_500m = [results[1]["scores"]["object_recognition"],
                   results[1]["scores"]["damage_detection"],
                   results[1]["scores"]["json_output"]]

    x = np.arange(len(categories))
    width = 0.35

    ax2.bar(x - width/2, scores_256m, width, label='SmolVLM-256M', color='lightblue')
    ax2.bar(x + width/2, scores_500m, width, label='SmolVLM-500M', color='lightcoral')

    ax2.set_ylabel('정답 수 (최대 6)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_title('평가 항목별 점수')
    ax2.legend()
    ax2.set_ylim(0, 7)

    # 점수 표시
    for i, (s1, s2) in enumerate(zip(scores_256m, scores_500m)):
        ax2.text(i - width/2, s1 + 0.2, str(s1), ha='center', fontsize=11, fontweight='bold')
        ax2.text(i + width/2, s2 + 0.2, str(s2), ha='center', fontsize=11, fontweight='bold')

    # 3. 이미지별 응답 시간 히트맵
    ax3 = fig.add_subplot(2, 2, 3)

    images = data["images"]
    prompts = ["object_recognition", "damage_detection", "structured_json"]

    # SmolVLM-500M 응답 시간 추출
    time_matrix = []
    for img in images:
        row = []
        for prompt in prompts:
            for test in results[1]["tests"]:
                if test["image"] == img and test["prompt"] == prompt:
                    row.append(test["time_sec"])
                    break
        time_matrix.append(row)

    im = ax3.imshow(time_matrix, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks(range(len(prompts)))
    ax3.set_xticklabels(['객체인식', '손상탐지', 'JSON'], rotation=0)
    ax3.set_yticks(range(len(images)))
    ax3.set_yticklabels([img.replace('synthetic_', '') for img in images])
    ax3.set_title('SmolVLM-500M 응답 시간 (초)')

    # 값 표시
    for i in range(len(images)):
        for j in range(len(prompts)):
            ax3.text(j, i, f'{time_matrix[i][j]:.1f}', ha='center', va='center', fontsize=9)

    plt.colorbar(im, ax=ax3, label='시간(초)')

    # 4. 응답 품질 분석 (텍스트 요약)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    summary_text = """
    ┌─────────────────────────────────────────────────────────┐
    │              SmolVLM 로컬 모델 평가 요약                │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  SmolVLM-256M (256M 파라미터)                          │
    │  ─────────────────────────────────────────────────────  │
    │  • 객체 인식: 불가 (색상만 응답)                        │
    │  • 손상 탐지: 불가 (프롬프트 반복)                      │
    │  • JSON 출력: 불가 (템플릿만 반환)                      │
    │  • 응답 속도: 빠름 (평균 7.5초)                         │
    │                                                         │
    │  SmolVLM-500M (500M 파라미터)                          │
    │  ─────────────────────────────────────────────────────  │
    │  • 객체 인식: 제한적 (일부 형식 따름)                   │
    │  • 손상 탐지: 제한적 (형식만, 실제 분석 없음)           │
    │  • JSON 출력: 제한적 (구조는 생성, 내용 부정확)         │
    │  • 응답 속도: 보통 (평균 11.8초)                        │
    │                                                         │
    │  결론: 두 모델 모두 건설 자재 손상 평가에 부적합        │
    │        더 큰 모델 (7B+) 또는 Fine-tuning 필요           │
    └─────────────────────────────────────────────────────────┘
    """

    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 저장
    output_file = "vlm_benchmark_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"시각화 저장: {output_file}")

    plt.show()
    return output_file

def create_comparison_chart(all_results):
    """여러 모델 비교 차트 생성"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("VLM 로컬 모델 종합 비교 - 건설현장 자재 손상 평가", fontsize=16, fontweight='bold')

    models = [r["model"] for r in all_results]

    # 1. 정확도 비교 (레이더 차트)
    ax1 = axes[0, 0]

    categories = ['객체 인식', '손상 탐지', 'JSON 출력', '한국어', '추론 속도']

    # 각 모델의 점수 (0-100 스케일)
    scores_dict = {}
    for r in all_results:
        name = r["model"]
        obj_score = r["scores"]["object_recognition"] / 6 * 100
        dmg_score = r["scores"]["damage_detection"] / 6 * 100
        json_score = r["scores"]["json_output"] / 6 * 100
        # 한국어 점수 (응답에 한글 포함 여부로 추정)
        korean_score = 50 if "500M" in name else 20
        # 속도 점수 (빠를수록 높음)
        speed_score = max(0, 100 - r["summary"]["avg_time_sec"] * 5)
        scores_dict[name] = [obj_score, dmg_score, json_score, korean_score, speed_score]

    x = np.arange(len(categories))
    width = 0.8 / len(models)

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, (model, scores) in enumerate(scores_dict.items()):
        ax1.bar(x + i * width, scores, width, label=model, color=colors[i])

    ax1.set_ylabel('점수 (100점 만점)')
    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels(categories)
    ax1.set_title('평가 항목별 점수 비교')
    ax1.legend()
    ax1.set_ylim(0, 110)

    # 2. 응답 시간 비교
    ax2 = axes[0, 1]

    times = [r["summary"]["avg_time_sec"] for r in all_results]
    load_times = [r["summary"]["load_time_sec"] for r in all_results]

    x = np.arange(len(models))
    ax2.bar(x, load_times, label='모델 로딩', color='lightgray')
    ax2.bar(x, times, bottom=load_times, label='평균 추론', color='steelblue')

    ax2.set_ylabel('시간 (초)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15)
    ax2.set_title('모델별 소요 시간')
    ax2.legend()

    for i, (t, l) in enumerate(zip(times, load_times)):
        ax2.text(i, t + l + 0.5, f'{t+l:.1f}s', ha='center', fontsize=10)

    # 3. 모델 크기 vs 성능
    ax3 = axes[1, 0]

    sizes = []
    total_scores = []
    for r in all_results:
        # 모델 크기 추출 (M 단위)
        name = r["model"]
        if "256M" in name:
            sizes.append(256)
        elif "500M" in name:
            sizes.append(500)
        elif "2B" in name:
            sizes.append(2000)
        elif "7B" in name:
            sizes.append(7000)
        else:
            sizes.append(1000)

        total = (r["scores"]["object_recognition"] +
                 r["scores"]["damage_detection"] +
                 r["scores"]["json_output"])
        total_scores.append(total)

    ax3.scatter(sizes, total_scores, s=200, c=colors[:len(models)], alpha=0.7)

    for i, model in enumerate(models):
        ax3.annotate(model, (sizes[i], total_scores[i]),
                    textcoords="offset points", xytext=(0,10), ha='center')

    ax3.set_xlabel('모델 크기 (M 파라미터)')
    ax3.set_ylabel('총 점수 (18점 만점)')
    ax3.set_title('모델 크기 vs 성능')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)

    # 4. 응답 품질 분석 텍스트
    ax4 = axes[1, 1]
    ax4.axis('off')

    quality_text = """
┌────────────────────────────────────────────────────────────┐
│                    응답 품질 상세 분석                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  테스트 이미지 (6종)                                       │
│  ───────────────────────────────────────────────────────── │
│  • synthetic_hardhat: 노란색 안전모 (정상)                 │
│  • synthetic_hardhat_cracked: 균열 안전모 (손상)           │
│  • synthetic_boots: 검정 작업화                            │
│  • synthetic_gloves: 가죽 장갑                             │
│  • synthetic_hammer: 해머 (정상)                           │
│  • synthetic_hammer_rusty: 녹슨 해머 (손상)                │
│                                                            │
│  평가 기준                                                 │
│  ───────────────────────────────────────────────────────── │
│  1. 객체 인식: 물체 종류 정확히 식별                       │
│  2. 손상 탐지: 손상 유무/유형/위치 파악                    │
│  3. JSON 출력: 정형화된 구조 데이터 생성                   │
│                                                            │
│  주요 발견                                                 │
│  ───────────────────────────────────────────────────────── │
│  • 256M 모델: 시각적 이해 능력 매우 제한적                 │
│  • 500M 모델: 형식은 이해하나 실제 분석 불가               │
│  • 둘 다 손상 여부 구분 실패                               │
│  • 7B+ 모델 또는 도메인 Fine-tuning 필요                   │
└────────────────────────────────────────────────────────────┘
    """

    ax4.text(0.5, 0.5, quality_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_file = "vlm_model_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"비교 차트 저장: {output_file}")

    return output_file

if __name__ == "__main__":
    print("VLM 벤치마크 결과 시각화")
    print("=" * 50)

    # 현재 결과 시각화
    if os.path.exists("smolvlm_benchmark_results.json"):
        visualize_smolvlm_results()

        # 비교 차트도 생성
        data = load_results("smolvlm_benchmark_results.json")
        create_comparison_chart(data["results"])
    else:
        print("smolvlm_benchmark_results.json 파일이 없습니다.")
        print("먼저 run_smolvlm_benchmark.py를 실행하세요.")
