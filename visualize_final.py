"""
VLM 벤치마크 최종 시각화
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_results():
    """결과 파일 로드"""
    results = []

    # 1. multi_vlm 결과
    if os.path.exists("multi_vlm_benchmark_results.json"):
        with open("multi_vlm_benchmark_results.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            for r in data["results"]:
                if "summary" in r and r["summary"]:
                    results.append(r)

    # 2. smolvlm 결과 (중복 방지)
    if os.path.exists("smolvlm_benchmark_results.json") and not results:
        with open("smolvlm_benchmark_results.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            for r in data["results"]:
                if "summary" in r:
                    results.append(r)

    return results

def create_comprehensive_visualization(results):
    """종합 시각화 생성"""

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("VLM 로컬 모델 성능 비교 분석\n건설현장 자재 손상 평가 벤치마크", fontsize=18, fontweight='bold')

    models = [r["model"] for r in results]
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(models), 3)))

    # ========================================
    # 1. 종합 점수 비교 (막대 차트)
    # ========================================
    ax1 = fig.add_subplot(2, 3, 1)

    categories = ['객체 인식', '손상 탐지', 'JSON 출력']
    x = np.arange(len(categories))
    width = 0.35

    for i, r in enumerate(results):
        scores = [
            r["scores"]["object_recognition"],
            r["scores"]["damage_detection"],
            r["scores"]["json_output"]
        ]
        offset = (i - len(results)/2 + 0.5) * width
        bars = ax1.bar(x + offset, scores, width, label=r["model"], color=colors[i])

        # 값 표시
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(score), ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel('점수 (최대 6)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.set_title('평가 항목별 점수', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 7.5)
    ax1.axhline(y=6, color='green', linestyle='--', alpha=0.5, label='만점')

    # ========================================
    # 2. 총점 비교 (수평 막대)
    # ========================================
    ax2 = fig.add_subplot(2, 3, 2)

    total_scores = [sum(r["scores"].values()) for r in results]
    y_pos = np.arange(len(models))

    bars = ax2.barh(y_pos, total_scores, color=colors[:len(models)], edgecolor='black')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models, fontsize=10)
    ax2.set_xlabel('총점 (최대 18)', fontsize=11)
    ax2.set_title('모델별 총점', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 20)
    ax2.axvline(x=18, color='green', linestyle='--', alpha=0.5)

    for bar, score in zip(bars, total_scores):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{score}/18 ({score/18*100:.0f}%)', va='center', fontsize=10, fontweight='bold')

    # ========================================
    # 3. 응답 시간 비교
    # ========================================
    ax3 = fig.add_subplot(2, 3, 3)

    load_times = [r["summary"]["load_time_sec"] for r in results]
    infer_times = [r["summary"]["avg_time_sec"] for r in results]

    x = np.arange(len(models))
    width = 0.6

    ax3.bar(x, load_times, width, label='모델 로딩', color='lightgray', edgecolor='gray')
    ax3.bar(x, infer_times, width, bottom=load_times, label='평균 추론', color='steelblue')

    ax3.set_ylabel('시간 (초)', fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=10)
    ax3.set_title('소요 시간', fontsize=12, fontweight='bold')
    ax3.legend()

    for i, (l, inf) in enumerate(zip(load_times, infer_times)):
        ax3.text(i, l + inf + 0.3, f'{l+inf:.1f}s', ha='center', fontsize=10, fontweight='bold')

    # ========================================
    # 4. 정확도 vs 속도 산점도
    # ========================================
    ax4 = fig.add_subplot(2, 3, 4)

    accuracy = [sum(r["scores"].values()) / 18 * 100 for r in results]
    speed = [r["summary"]["avg_time_sec"] for r in results]
    sizes = [r.get("size_mb", 500) / 10 for r in results]

    scatter = ax4.scatter(speed, accuracy, s=[s*5 for s in sizes], c=colors[:len(models)],
                          alpha=0.7, edgecolors='black', linewidths=2)

    for i, model in enumerate(models):
        ax4.annotate(model, (speed[i], accuracy[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=9, fontweight='bold')

    ax4.set_xlabel('평균 추론 시간 (초)', fontsize=11)
    ax4.set_ylabel('정확도 (%)', fontsize=11)
    ax4.set_title('정확도 vs 속도', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 이상적 영역 표시 (빠르고 정확)
    ax4.axhspan(50, 100, xmin=0, xmax=0.5, alpha=0.1, color='green')
    ax4.text(3, 90, '이상적 영역\n(빠르고 정확)', fontsize=9, color='green')

    # ========================================
    # 5. 성공률 파이 차트
    # ========================================
    ax5 = fig.add_subplot(2, 3, 5)

    success_rates = [r["summary"]["success_rate"] for r in results]

    # 막대 차트로 변경 (파이보다 비교 용이)
    bars = ax5.bar(models, success_rates, color=colors[:len(models)], edgecolor='black')

    ax5.set_ylabel('성공률 (%)', fontsize=11)
    ax5.set_title('추론 성공률', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 110)

    for bar, rate in zip(bars, success_rates):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.0f}%', ha='center', fontsize=11, fontweight='bold')

    # ========================================
    # 6. 상세 결과 테이블
    # ========================================
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # 테이블 데이터 구성
    table_data = []
    for r in results:
        total = sum(r["scores"].values())
        table_data.append([
            r["model"],
            f'{r.get("size_mb", "N/A")} MB',
            f'{r["summary"]["load_time_sec"]:.1f}s',
            f'{r["summary"]["avg_time_sec"]:.1f}s',
            f'{r["scores"]["object_recognition"]}/6',
            f'{r["scores"]["damage_detection"]}/6',
            f'{r["scores"]["json_output"]}/6',
            f'{total}/18',
            f'{total/18*100:.0f}%'
        ])

    headers = ['모델', '크기', '로딩', '추론', '객체', '손상', 'JSON', '총점', '정확도']

    table = ax6.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['lightsteelblue'] * len(headers)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)

    # 헤더 스타일
    for i in range(len(headers)):
        table[(0, i)].set_text_props(fontweight='bold')

    ax6.set_title('상세 결과 요약', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # 저장
    output_file = "vlm_comprehensive_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"저장 완료: {output_file}")

    return output_file

def create_test_images_overview():
    """테스트 이미지 개요 생성"""
    from PIL import Image

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("테스트 이미지 (test_images/ 폴더)", fontsize=14, fontweight='bold')

    test_images = [
        ("synthetic_hardhat.jpg", "안전모 (정상)"),
        ("synthetic_hardhat_cracked.jpg", "안전모 (균열)"),
        ("synthetic_boots.jpg", "안전화"),
        ("synthetic_gloves.jpg", "작업장갑"),
        ("synthetic_hammer.jpg", "해머 (정상)"),
        ("synthetic_hammer_rusty.jpg", "해머 (녹)")
    ]

    for idx, (filename, title) in enumerate(test_images):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        filepath = f"test_images/{filename}"
        if os.path.exists(filepath):
            img = Image.open(filepath)
            ax.imshow(img)
            ax.set_title(f"{title}\n({filename})", fontsize=10)
        else:
            ax.text(0.5, 0.5, f"이미지 없음\n{filename}", ha='center', va='center')
            ax.set_title(title)

        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("test_images_overview.png", dpi=150, bbox_inches='tight', facecolor='white')
    print("저장 완료: test_images_overview.png")

def main():
    print("=" * 60)
    print("VLM 벤치마크 최종 시각화")
    print("=" * 60)

    # 결과 로드
    results = load_results()

    if not results:
        print("결과 파일을 찾을 수 없습니다.")
        return

    print(f"로드된 모델: {len(results)}개")
    for r in results:
        print(f"  - {r['model']}")

    # 시각화 생성
    print("\n시각화 생성 중...")
    create_comprehensive_visualization(results)

    # 테스트 이미지 개요
    print("테스트 이미지 개요 생성 중...")
    create_test_images_overview()

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
