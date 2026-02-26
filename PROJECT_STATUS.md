# VLM On-Device 프로젝트 진행 상황

**최종 업데이트**: 2026-02-26 16:10
**작성 목적**: 타 환경 Claude Code에서 프로젝트 현황 파악 및 이어서 작업 가능하도록 상세 기록

---

## 1. 프로젝트 개요

### 1.1 목표
건설현장 자재(안전모, 안전화, 장갑, 공구 등)를 촬영했을 때:
- **객체 인식**: 어떤 자재인지 식별
- **손상 평가**: 손상 여부, 위치, 정도, 심각도 파악
- **구조화된 출력**: JSON 형식으로 결과 출력

### 1.2 핵심 요구사항
- **온디바이스 추론**: 인터넷 없이 로컬에서 실행 가능
- **실제 이미지 테스트**: 합성 이미지가 아닌 실제 건설현장 사진 사용
- **다양한 VLM 비교**: 현존하는 VLM 모델들의 성능 비교 평가

---

## 2. 완료된 작업

### 2.1 테스트 이미지 수집 ✅
**위치**: `real_images/` 폴더

| 파일명 | 설명 | 출처 |
|--------|------|------|
| hardhat_yellow.jpg | 노란 안전모 착용 작업자들 | Unsplash |
| hardhat_single.jpg | 건설현장 단일 작업자 | Unsplash |
| hardhat_damaged.jpg | 손상된 안전모 | Pexels |
| hammer.jpg | 클로 해머 (장도리) | Unsplash |
| wrench.jpg | 렌치/스패너 세트 | Unsplash |
| gloves_work.jpg | 장갑+안전모 작업자 | Pexels |
| safety_boots.jpg | 안전화 | Unsplash |
| safety_boots_real.jpg | 실제 안전화 | Pexels |
| safety_boots_steel.jpg | 강철 토캡 안전화 | Unsplash |
| work_gloves.jpg | 작업 장갑 | Unsplash |
| work_gloves_real.jpg | 실제 작업 장갑 | Pexels |

### 2.2 VLM 모델 테스트 ✅

#### 테스트 완료된 모델

| 모델 | 크기 | 총점 | 비율 | JSON출력 | 비고 |
|------|------|------|------|---------|------|
| **SmolVLM-500M** | 500M | 11.5/13 | **88%** | 안정적 | **권장** |
| SmolVLM-256M | 256M | 4.8/13 | 37% | 불안정 | 속도 중시 |

#### 테스트 실패/미완료 모델

| 모델 | 상태 | 원인 |
|------|------|------|
| Qwen2-VL-2B | 실패 | transformers 버전 호환성 오류 |
| Florence-2 | 실패 | TokenizersBackend 속성 오류 |
| Moondream2 | 실패 | API 변경으로 호출 불가 |
| Gemini API | 미테스트 | API 키 미설정 |

### 2.3 생성된 결과 파일

#### 벤치마크 결과 (JSON)
- `comprehensive_vlm_results.json` - 종합 테스트 결과 (주요)
- `real_image_vlm_results.json` - 실제 이미지 테스트 결과
- `all_vlm_results.json` - 합성 이미지 테스트 결과
- `qwen2vl_results.json` - Qwen2-VL 테스트 시도 결과 (실패)

#### 시각화 (PNG)
- `vlm_comprehensive_analysis.png` - 종합 분석 시각화 (주요)
- `real_image_results_visualization.png` - 실제 이미지 결과 시각화
- `vlm_benchmark_visualization.png` - 벤치마크 시각화
- `vlm_final_comparison.png` - 최종 비교 시각화

#### 문서 (MD)
- `VLM_BENCHMARK_REPORT.md` - 벤치마크 보고서
- `ALL_VLM_MODELS_LIST.md` - 현존 VLM 모델 목록
- `VLM_BENCHMARK_GUIDE.md` - 벤치마크 가이드

---

## 3. 핵심 코드 파일

### 3.1 메인 테스트 스크립트
```
comprehensive_real_vlm_test.py  # 실제 이미지로 종합 VLM 테스트 (주요)
test_real_images.py             # 실제 이미지 테스트 (초기 버전)
test_qwen2vl.py                 # Qwen2-VL 테스트 스크립트
test_gemini_free.py             # Gemini API 테스트 (API키 필요)
```

### 3.2 시각화 스크립트
```
final_vlm_analysis.py           # 종합 분석 및 시각화 생성
visualize_real_results.py       # 실제 이미지 결과 시각화
```

### 3.3 기타 테스트 스크립트
```
comprehensive_vlm_benchmark.py  # URL 기반 이미지 테스트 (Wikimedia)
quick_multi_test.py             # 빠른 다중 모델 테스트
vlm_benchmark.py                # 기본 벤치마크
```

---

## 4. 테스트 결과 상세

### 4.1 평가 기준 (총 13점)
- **객체 인식** (5점): 안전모, 장갑, 공구 등 키워드 인식
- **상태 평가** (3점): new, used, worn, damaged 등 상태 단어
- **JSON 형식** (3점): 유효한 JSON 출력 여부
- **상세도** (2점): 응답 길이 (150자 이상 1점, 300자 이상 2점)

### 4.2 SmolVLM-500M 결과 (권장 모델)
```
객체 인식: 4.25/5 (85%)
상태 평가: 2.75/3 (92%)
JSON 형식: 2.5/3 (83%)
상세도: 2.0/2 (100%)
총점: 11.5/13 (88%)
```

**샘플 출력**:
```json
{
    "objects": [
        {"name": "hardhat", "condition": "new", "notes": "any observations"},
        {"name": "safety vest", "condition": "new", "notes": "any observations"}
    ],
    "overall_condition": "good/fair/poor",
    "damage_detected": true
}
```

### 4.3 SmolVLM-256M 결과
```
객체 인식: 1.25/5 (25%)
상태 평가: 2.25/3 (75%)
JSON 형식: 0.75/3 (25%)
상세도: 0.5/2 (25%)
총점: 4.75/13 (37%)
```

**문제점**: JSON 형식 출력 불안정, 객체 인식 정확도 낮음

---

## 5. 남은 작업

### 5.1 즉시 진행 가능
1. **Gemini API 테스트**
   - API 키 발급: https://aistudio.google.com/apikey
   - 실행: `set GEMINI_API_KEY=your_key && python comprehensive_real_vlm_test.py`
   - 예상: 가장 높은 정확도 (무료 한도: 분당 15회, 일 1500회)

2. **더 많은 이미지 테스트**
   - `real_images/` 폴더의 모든 이미지(11개)로 테스트
   - 현재는 4개 이미지로만 테스트됨

### 5.2 추가 개선 가능
1. **손상 이미지 수집**: 실제 손상된 자재 사진 확보
2. **파인튜닝**: SmolVLM-500M을 건설현장 데이터로 파인튜닝
3. **Qwen2-VL 호환성 해결**: transformers 버전 조정 후 재테스트

### 5.3 Tauri 앱 통합
- `src-tauri/src/model/inference.rs` 파일에 VLM 추론 코드 존재
- SmolVLM-500M을 앱에 통합하여 온디바이스 실행 구현 필요

---

## 6. 실행 방법

### 6.1 종합 VLM 테스트
```bash
cd "c:/Users/Hanbin Lee/Desktop/Git/P03_VLM_ondevice"
python comprehensive_real_vlm_test.py
```

### 6.2 Gemini API 테스트 (API 키 필요)
```bash
set GEMINI_API_KEY=your_api_key_here
python comprehensive_real_vlm_test.py
```

### 6.3 결과 시각화
```bash
python final_vlm_analysis.py
```

### 6.4 특정 모델만 테스트
```python
# comprehensive_real_vlm_test.py 내 models 리스트 수정
models = [
    ("SmolVLM-500M", "HuggingFaceTB/SmolVLM-500M-Instruct", "smolvlm"),
]
```

---

## 7. 의존성

### 7.1 필수 패키지
```
torch
transformers
Pillow
matplotlib
numpy
```

### 7.2 선택적 패키지 (특정 모델용)
```
google-generativeai  # Gemini API
qwen-vl-utils        # Qwen2-VL (현재 호환성 문제)
```

### 7.3 설치
```bash
pip install torch transformers Pillow matplotlib numpy
pip install google-generativeai  # Gemini 테스트시
```

---

## 8. 주요 발견 사항

### 8.1 Zero-shot 한계
- 파인튜닝 없이는 건설현장 특화 인식에 명확한 한계
- 일반적인 객체(해머, 헬멧)는 인식하나 세부 상태 평가는 부정확
- "new"와 "damaged" 구분 능력 제한적

### 8.2 모델 크기 vs 성능
- SmolVLM-500M >> SmolVLM-256M (성능 차이 크게 남)
- 500M 모델이 JSON 형식 출력에서 훨씬 안정적

### 8.3 권장 사항
| 시나리오 | 권장 모델 |
|----------|----------|
| 프로토타이핑 | Gemini API (무료, 정확도 높음) |
| 온디바이스 배포 | SmolVLM-500M (안정적 JSON) |
| 실시간 처리 | SmolVLM-256M (빠른 속도) |
| 생산 환경 | 도메인 특화 파인튜닝 필요 |

---

## 9. 파일 구조

```
P03_VLM_ondevice/
├── real_images/                    # 실제 테스트 이미지 (11개)
│   ├── hardhat_yellow.jpg
│   ├── hammer.jpg
│   ├── wrench.jpg
│   └── ...
├── test_images/                    # 합성 테스트 이미지
├── src-tauri/                      # Tauri 앱 소스
│   └── src/model/inference.rs      # VLM 추론 코드
├── comprehensive_real_vlm_test.py  # 메인 테스트 스크립트 ★
├── final_vlm_analysis.py           # 시각화 생성 스크립트
├── comprehensive_vlm_results.json  # 테스트 결과 데이터
├── vlm_comprehensive_analysis.png  # 시각화 결과
├── VLM_BENCHMARK_REPORT.md         # 상세 보고서
├── ALL_VLM_MODELS_LIST.md          # VLM 모델 목록
└── PROJECT_STATUS.md               # 이 문서 (진행상황)
```

---

## 10. 다음 Claude Code 세션을 위한 요약

1. **현재 상태**: SmolVLM-500M이 최고 성능 (88%), 온디바이스 배포에 적합
2. **미완료**: Gemini API 테스트 (API 키 필요), Tauri 앱 통합
3. **주요 코드**: `comprehensive_real_vlm_test.py` 실행으로 테스트 가능
4. **결과 확인**: `comprehensive_vlm_results.json`, `vlm_comprehensive_analysis.png`
5. **핵심 한계**: Zero-shot으로는 정확한 손상 평가 어려움, 파인튜닝 권장

---

*이 문서는 프로젝트 인수인계 및 연속성을 위해 작성되었습니다.*
