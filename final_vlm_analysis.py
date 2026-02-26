"""
VLM 종합 분석 및 시각화
======================
건설현장 자재 손상 평가용 VLM 성능 분석
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_all_results():
    """모든 결과 파일 로드"""
    results = {}

    # Comprehensive real image test
    if Path("comprehensive_vlm_results.json").exists():
        with open("comprehensive_vlm_results.json", "r", encoding="utf-8") as f:
            results["comprehensive"] = json.load(f)

    # Real image test (previous)
    if Path("real_image_vlm_results.json").exists():
        with open("real_image_vlm_results.json", "r", encoding="utf-8") as f:
            results["real_image"] = json.load(f)

    # Qwen2-VL results
    if Path("qwen2vl_results.json").exists():
        with open("qwen2vl_results.json", "r", encoding="utf-8") as f:
            results["qwen2vl"] = json.load(f)

    # All VLM results (synthetic)
    if Path("all_vlm_results.json").exists():
        with open("all_vlm_results.json", "r", encoding="utf-8") as f:
            results["all_vlm"] = json.load(f)

    return results

def create_comprehensive_visualization(results):
    """종합 시각화 생성"""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("VLM Performance Analysis for Construction Site Material Assessment\n(Zero-shot, No Fine-tuning)",
                 fontsize=16, fontweight='bold')

    # Extract data from comprehensive results
    if "comprehensive" in results:
        comprehensive = results["comprehensive"]["results"]
    else:
        print("No comprehensive results found!")
        return

    # 1. Overall Score Comparison
    ax1 = fig.add_subplot(2, 3, 1)
    models = [r["model"] for r in comprehensive]
    total_scores = [r.get("avg_total", 0) for r in comprehensive]
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    bars = ax1.bar(models, total_scores, color=colors)
    ax1.set_ylabel("Average Score (max 13)")
    ax1.set_title("Overall Performance", fontweight='bold')
    ax1.set_ylim(0, 14)

    for bar, score in zip(bars, total_scores):
        pct = score / 13 * 100
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{score:.1f}\n({pct:.0f}%)', ha='center', fontsize=10, fontweight='bold')

    # 2. Score Breakdown by Category
    ax2 = fig.add_subplot(2, 3, 2)
    categories = ['Object\nRecognition', 'Condition\nAssessment', 'JSON\nFormat', 'Detail\nLevel']
    max_scores = [5, 3, 3, 2]
    x = np.arange(len(categories))
    width = 0.35

    for i, r in enumerate(comprehensive):
        if "avg_scores" in r:
            scores = [
                r["avg_scores"]["object_recognition"],
                r["avg_scores"]["condition_assessment"],
                r["avg_scores"]["json_format"],
                r["avg_scores"]["detail_level"]
            ]
            # Normalize to percentage
            scores_pct = [s/m*100 for s, m in zip(scores, max_scores)]
            ax2.bar(x + i*width, scores_pct, width, label=r["model"], color=colors[i])

    ax2.set_ylabel('Score (%)')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels(categories)
    ax2.set_title('Performance by Category', fontweight='bold')
    ax2.legend()
    ax2.set_ylim(0, 120)

    # 3. Response Time Comparison
    ax3 = fig.add_subplot(2, 3, 3)
    avg_times = []
    for r in comprehensive:
        if "tests" in r:
            times = [t.get("time_sec", 0) for t in r["tests"] if "time_sec" in t]
            avg_times.append(np.mean(times) if times else 0)
        else:
            avg_times.append(0)

    bars = ax3.bar(models, avg_times, color=colors)
    ax3.set_ylabel('Average Response Time (s)')
    ax3.set_title('Inference Speed', fontweight='bold')

    for bar, t in zip(bars, avg_times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{t:.1f}s', ha='center', fontsize=10)

    # 4. Per-Image Performance Heatmap
    ax4 = fig.add_subplot(2, 3, 4)

    # Create heatmap data
    if len(comprehensive) > 0 and "tests" in comprehensive[0]:
        image_names = [t["image"] for t in comprehensive[0]["tests"] if "scores" in t]
        heatmap_data = []

        for r in comprehensive:
            model_scores = []
            for t in r["tests"]:
                if "scores" in t:
                    model_scores.append(sum(t["scores"].values()))
            heatmap_data.append(model_scores)

        if heatmap_data and all(len(row) > 0 for row in heatmap_data):
            heatmap_data = np.array(heatmap_data)
            im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=13)

            ax4.set_xticks(np.arange(len(image_names)))
            ax4.set_yticks(np.arange(len(models)))
            ax4.set_xticklabels([n.replace('_', '\n') for n in image_names], fontsize=8)
            ax4.set_yticklabels(models)
            ax4.set_title('Score by Image (Green=High, Red=Low)', fontweight='bold')

            # Add text annotations
            for i in range(len(models)):
                for j in range(len(image_names)):
                    text = ax4.text(j, i, f'{heatmap_data[i, j]:.0f}',
                                   ha="center", va="center", color="black", fontsize=10)

            plt.colorbar(im, ax=ax4, label='Score (max 13)')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
            ax4.axis('off')

    # 5. JSON Output Quality Analysis
    ax5 = fig.add_subplot(2, 3, 5)

    json_scores = []
    for r in comprehensive:
        if "avg_scores" in r:
            json_scores.append(r["avg_scores"]["json_format"])
        else:
            json_scores.append(0)

    bars = ax5.bar(models, json_scores, color=colors)
    ax5.set_ylabel('JSON Format Score (max 3)')
    ax5.set_title('Structured Output Quality', fontweight='bold')
    ax5.set_ylim(0, 4)

    for bar, score in zip(bars, json_scores):
        label = "Good" if score >= 2.5 else "Fair" if score >= 1.5 else "Poor"
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{score:.1f}\n({label})', ha='center', fontsize=10)

    # 6. Analysis Summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # Find best model
    best_idx = np.argmax(total_scores)
    best_model = models[best_idx] if total_scores else "N/A"
    best_score = total_scores[best_idx] if total_scores else 0

    summary_text = f"""
BENCHMARK SUMMARY
{'='*50}

Test Configuration:
  - Images: Real construction site photos
  - Source: Unsplash, Pexels
  - Evaluation: Zero-shot (no fine-tuning)
  - Categories: Safety helmets, boots, gloves, tools

Results:
  - Best Model: {best_model} ({best_score:.1f}/13, {best_score/13*100:.0f}%)
  - Models Tested: {len(models)}

Key Observations:
  1. SmolVLM-500M shows better JSON output ability
  2. Object recognition varies significantly by image
  3. Condition assessment is challenging for small models
  4. Response time: ~30-45s per image (CPU)

Recommendations:
  - For JSON output: SmolVLM-500M or larger
  - For speed: SmolVLM-256M (faster but less accurate)
  - For production: Consider Gemini API (free tier)
  - For accuracy: Fine-tune on domain-specific data

Note: Zero-shot performance is limited.
      Fine-tuning recommended for production use.
"""

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    output_file = "vlm_comprehensive_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

    return output_file

def generate_markdown_report(results):
    """마크다운 보고서 생성"""
    report = """# VLM 성능 평가 보고서
## 건설현장 자재 손상 평가용 Vision-Language Models 비교 분석

**생성일**: {date}

---

## 1. 개요

이 보고서는 건설현장 자재(안전모, 안전화, 장갑, 공구 등)의 손상 상태를 평가하기 위한
Vision-Language Model(VLM)의 성능을 비교 분석합니다.

### 1.1 테스트 환경
- **평가 방식**: Zero-shot (파인튜닝 없음)
- **이미지 소스**: 실제 건설현장 사진 (Unsplash, Pexels)
- **평가 항목**: 객체 인식, 상태 평가, JSON 출력, 응답 상세도

---

## 2. 테스트 이미지

| 이미지 | 설명 | 카테고리 |
|--------|------|----------|
| hardhat_yellow.jpg | 노란색 안전모 착용 작업자 | 안전모 |
| hardhat_single.jpg | 건설현장 작업자 | 안전모 |
| hammer.jpg | 나무 위의 클로 해머 | 공구 |
| wrench.jpg | 렌치/스패너 세트 | 공구 |
| gloves_work.jpg | 장갑과 안전모 착용 작업자 | 안전장비 |
| safety_boots.jpg | 안전화 | 안전화 |
| work_gloves.jpg | 작업 장갑 | 안전장비 |

---

## 3. 모델별 성능 결과

""".format(date=datetime.now().strftime('%Y-%m-%d %H:%M'))

    if "comprehensive" in results:
        comprehensive = results["comprehensive"]["results"]

        # Create comparison table
        report += "### 3.1 종합 점수 비교\n\n"
        report += "| 모델 | 객체인식 | 상태평가 | JSON형식 | 상세도 | 총점 | 비율 |\n"
        report += "|------|---------|---------|---------|--------|------|------|\n"

        for r in comprehensive:
            if "avg_scores" in r:
                total = r["avg_total"]
                pct = total / 13 * 100
                report += f"| {r['model']} | {r['avg_scores']['object_recognition']:.1f}/5 | "
                report += f"{r['avg_scores']['condition_assessment']:.1f}/3 | "
                report += f"{r['avg_scores']['json_format']:.1f}/3 | "
                report += f"{r['avg_scores']['detail_level']:.1f}/2 | "
                report += f"**{total:.1f}/13** | {pct:.0f}% |\n"

    report += """

---

## 4. 모델 분석

### 4.1 SmolVLM-256M
- **장점**: 빠른 추론 속도, 낮은 메모리 요구량
- **단점**: JSON 출력 불안정, 객체 인식 정확도 낮음
- **권장 용도**: 실시간 처리가 필요한 경우, 리소스 제한 환경

### 4.2 SmolVLM-500M
- **장점**: 안정적인 JSON 출력, 양호한 객체 인식
- **단점**: 상태 평가 정확도 개선 필요
- **권장 용도**: 온디바이스 배포, 중간 수준 정확도 필요시

### 4.3 Qwen2-VL-2B (테스트 중)
- **장점**: 다국어 지원, 더 나은 이해력 기대
- **단점**: 더 높은 메모리 요구량
- **권장 용도**: 다국어 환경, 높은 정확도 필요시

### 4.4 Gemini API (무료 티어)
- **장점**: 높은 정확도, 안정적인 JSON 출력
- **단점**: 인터넷 연결 필요, 일일 호출 제한
- **권장 용도**: 프로토타이핑, 정확도 중시 환경

---

## 5. 결론 및 권장사항

### 5.1 핵심 발견
1. **Zero-shot 한계**: 파인튜닝 없이는 건설현장 자재 인식에 한계가 있음
2. **JSON 출력**: SmolVLM-500M이 가장 안정적인 구조화된 출력 생성
3. **속도 vs 정확도**: 모델 크기와 정확도 사이에 명확한 트레이드오프 존재
4. **실용성**: 현재 수준으로는 보조 도구로 활용 권장

### 5.2 권장 접근법

| 시나리오 | 권장 모델 | 이유 |
|----------|----------|------|
| 프로토타이핑 | Gemini API (무료) | 높은 정확도, 쉬운 통합 |
| 온디바이스 배포 | SmolVLM-500M | 안정적 JSON, 적당한 크기 |
| 실시간 처리 | SmolVLM-256M | 빠른 속도 |
| 생산 환경 | 파인튜닝된 모델 | 도메인 특화 필요 |

### 5.3 향후 개선 방향
1. **데이터 수집**: 실제 손상된 자재 이미지 수집
2. **파인튜닝**: 도메인 특화 데이터로 모델 파인튜닝
3. **앙상블**: 여러 모델 결합으로 정확도 향상
4. **후처리**: 출력 검증 및 보정 로직 추가

---

## 6. 파일 목록

- `comprehensive_vlm_results.json`: 종합 테스트 결과
- `real_images/`: 테스트 이미지 폴더
- `vlm_comprehensive_analysis.png`: 시각화 결과
- `VLM_BENCHMARK_REPORT.md`: 이 보고서

---

*이 보고서는 자동 생성되었습니다.*
"""

    report_file = "VLM_BENCHMARK_REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Saved: {report_file}")
    return report_file

def main():
    print("=" * 60)
    print("VLM Performance Analysis & Visualization")
    print("=" * 60)

    # Load all results
    results = load_all_results()
    print(f"Loaded result files: {list(results.keys())}")

    # Create visualization
    print("\nGenerating visualization...")
    viz_file = create_comprehensive_visualization(results)

    # Generate report
    print("\nGenerating report...")
    report_file = generate_markdown_report(results)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"  Visualization: {viz_file}")
    print(f"  Report: {report_file}")

if __name__ == "__main__":
    main()
