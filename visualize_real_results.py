"""
실제 이미지 VLM 테스트 결과 시각화
"""
import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def main():
    # 결과 로드
    with open("real_image_vlm_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("VLM Real Image Test Results\nConstruction Equipment Recognition (Zero-shot, No Fine-tuning)",
                 fontsize=16, fontweight='bold')

    models = [r["model"] for r in results]
    colors = ['#4CAF50', '#2196F3']

    # 1. Average Scores by Category
    ax1 = fig.add_subplot(2, 2, 1)

    categories = ['Object\nRecognition', 'Condition\nAssessment', 'JSON\nFormat', 'Detail\nLevel']
    max_scores = [5, 3, 3, 2]

    x = np.arange(len(categories))
    width = 0.35

    for i, r in enumerate(results):
        scores = [
            r["avg_scores"]["object_recognition"],
            r["avg_scores"]["condition_assessment"],
            r["avg_scores"]["json_format"],
            r["avg_scores"]["detail_level"]
        ]
        # Normalize to percentage
        scores_pct = [s/m*100 for s, m in zip(scores, max_scores)]
        bars = ax1.bar(x + i*width, scores_pct, width, label=r["model"], color=colors[i])

        for bar, s, m in zip(bars, scores, max_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{s:.1f}/{m}', ha='center', fontsize=9)

    ax1.set_ylabel('Score (%)')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels(categories)
    ax1.set_title('Average Scores by Category', fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, 120)

    # 2. Total Score Comparison
    ax2 = fig.add_subplot(2, 2, 2)

    total_scores = []
    for r in results:
        total = sum(r["avg_scores"].values())
        total_scores.append(total)

    max_total = 13  # 5+3+3+2
    bars = ax2.bar(models, total_scores, color=colors)

    for bar, score in zip(bars, total_scores):
        pct = score / max_total * 100
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{score:.1f}/13\n({pct:.0f}%)', ha='center', fontsize=11, fontweight='bold')

    ax2.set_ylabel('Total Score')
    ax2.set_title('Total Score Comparison', fontweight='bold')
    ax2.set_ylim(0, 15)

    # 3. Per-Image Performance (SmolVLM-256M)
    ax3 = fig.add_subplot(2, 2, 3)

    r256 = results[0]
    images = [t["image"].replace("_", "\n") for t in r256["tests"]]
    img_scores = [sum(t["scores"].values()) for t in r256["tests"]]

    bars = ax3.barh(images, img_scores, color='#4CAF50')
    ax3.set_xlabel('Score (max 13)')
    ax3.set_title('SmolVLM-256M: Per-Image Scores', fontweight='bold')
    ax3.set_xlim(0, 15)

    for bar, score in zip(bars, img_scores):
        ax3.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{score}', va='center', fontweight='bold')

    # 4. Response Quality Analysis
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    analysis_text = """
REAL IMAGE TEST RESULTS ANALYSIS
================================

Test Images (from Unsplash/Pexels):
  - Construction site with workers wearing hardhats
  - Construction workers with safety vests
  - Claw hammer on wooden surface
  - Wrench/spanner set on wall
  - Worker wearing gloves and hardhat

SmolVLM-256M Results:
  - Object Recognition: 2.8/5 (56%) - Detected hammer, wrench, construction
  - Condition Assessment: 1.2/3 (40%) - Mentioned "new", "worn", "damage"
  - JSON Format: 0/3 (0%) - Failed to output valid JSON
  - Detail Level: 1.6/2 (80%) - Generated detailed descriptions
  - TOTAL: 5.6/13 (43%)

SmolVLM-500M Results:
  - Object Recognition: 0.4/5 (8%) - Poor object identification
  - Condition Assessment: 0.8/3 (27%) - Basic condition words only
  - JSON Format: 0.6/3 (20%) - Partial JSON attempt
  - Detail Level: 0.4/2 (20%) - Very short responses
  - TOTAL: 2.2/13 (17%)

KEY FINDINGS:
  1. SmolVLM-256M performed BETTER than 500M on real images
  2. Neither model produced valid JSON output
  3. Object recognition is inconsistent
  4. Models understand "construction" context but fail specifics
  5. Zero-shot is insufficient for this task

RECOMMENDATION:
  - Use Gemini API (free) for accurate results
  - Or fine-tune on construction equipment dataset
"""

    ax4.text(0.05, 0.95, analysis_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    output = "real_image_results_visualization.png"
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output}")

if __name__ == "__main__":
    main()
