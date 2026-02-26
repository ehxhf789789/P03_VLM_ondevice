# VLM 성능 평가 결과 보고서
## 건설현장 자재 손상 평가 프로토타입

**평가일**: 2026-02-26
**평가 목적**: 건설현장 자재(안전모, 안전화, 장갑, 공구)의 손상 인식 및 정형 데이터 출력 능력 평가

---

## 1. 평가 개요

### 1.1 테스트 이미지 기준

| 카테고리 | 테스트 이미지 | 설명 |
|---------|-------------|------|
| 안전모 | `synthetic_hardhat` | 노란색 안전모 (정상) |
| 안전모 | `synthetic_hardhat_cracked` | 균열 있는 안전모 (손상) |
| 안전화 | `synthetic_boots` | 검정색 작업화 (정상) |
| 장갑 | `synthetic_gloves` | 가죽 작업 장갑 (정상) |
| 공구 | `synthetic_hammer` | 해머 (정상) |
| 공구 | `synthetic_hammer_rusty` | 녹슨 해머 (손상) |

### 1.2 평가 프롬프트

1. **객체 인식 (Object Recognition)**: 물체 이름, 유형, 색상 식별
2. **손상 탐지 (Damage Detection)**: 손상 유무, 유형, 위치, 심각도 분석
3. **정형 JSON 출력**: 구조화된 JSON 형식으로 결과 출력

---

## 2. 현존 VLM 모델 목록

### 2.1 로컬 실행 모델 (온디바이스)

| 모델 | 크기 | GPU 메모리 | 특징 | 추천 용도 |
|------|-----|----------|------|----------|
| **SmolVLM-256M** | 256M | ~500MB | 최소형 | 모바일 (제한적) |
| **SmolVLM-500M** | 500M | ~800MB | 균형형 | 엣지 디바이스 |
| **SmolVLM-2B** | 2B | ~3GB | 경량 VLM | 일반 PC |
| **Moondream2** | 1.6B | ~2GB | 빠른 추론 | 프로토타입 |
| **Qwen2-VL-2B** | 2B | ~3GB | OCR 강점 | 문서/텍스트 |
| **PaliGemma-3B** | 3B | ~4GB | Google | 다국어 |
| **Qwen2.5-VL-7B** | 7B | ~10GB | 최신 고성능 | 서버 |
| **LLaVA-1.6-7B** | 7B | ~10GB | 범용 | 서버 |
| **InternVL2-8B** | 8B | ~12GB | 최고 성능 | 서버 |

### 2.2 클라우드 API 모델

| 모델 | 제공사 | 비용 | 무료 한도 | 추천 |
|------|-------|-----|---------|-----|
| **Gemini 2.0 Flash** | Google | 무료 | 15 req/min, 1500/day | **프로토타입 최적** |
| **Gemini 1.5 Pro** | Google | 무료/유료 | 2 req/min | 고성능 필요시 |
| **GPT-4o-mini** | OpenAI | $0.15/1M | 없음 | 가성비 |
| **GPT-4o** | OpenAI | $5/1M | 없음 | 최고 성능 |
| **Claude 3 Haiku** | Anthropic | $0.25/1M | 없음 | 빠른 응답 |
| **Claude 3.5 Sonnet** | Anthropic | $3/1M | 없음 | 코드 생성 |

---

## 3. SmolVLM 로컬 모델 테스트 결과

### 3.1 테스트 환경
- **디바이스**: CPU (Windows 11)
- **테스트 이미지**: 6개 (합성 이미지)
- **프롬프트**: 3종류

### 3.2 정량적 결과

| 모델 | 로딩 시간 | 평균 응답 | 성공률 | 객체 인식 | 손상 탐지 | JSON 출력 |
|------|---------|---------|-------|---------|---------|----------|
| **SmolVLM-256M** | 6.17s | 7.51s | 100% | 2/6 | 2/6 | 1/6 |
| **SmolVLM-500M** | 5.61s | 11.8s | 100% | 5/6 | 6/6 | 3/6 |

### 3.3 정성적 분석

#### SmolVLM-256M
```
문제점:
- 객체를 정확히 인식하지 못함 ("Dark Brown", "Blue" 등 색상만 응답)
- 프롬프트를 그대로 반복하는 경향
- JSON 구조 생성 능력 미흡
- 한국어 이해/생성 능력 부족

예시 응답 (안전모 이미지):
- "The circle is blue." (실제: 노란색 안전모)
- 균열 감지 실패
```

#### SmolVLM-500M
```
개선점:
- 프롬프트 형식 일부 따름
- JSON 구조 생성 시도
- 한국어 키워드 일부 포함

문제점:
- 실제 객체 인식 실패 ("무지", "무성" 등 엉뚱한 응답)
- 손상 상태 분석 불가
- 템플릿만 반복 (실제 분석 없음)

예시 응답:
- "물체 이름: 무지\n- 유형: 안전모" (이미지와 무관한 답변)
```

### 3.4 결론

| 평가 항목 | SmolVLM-256M | SmolVLM-500M | 평가 |
|---------|-------------|-------------|------|
| 객체 인식 | 불가 | 불가 | **부적합** |
| 손상 탐지 | 불가 | 불가 | **부적합** |
| JSON 출력 | 제한적 | 제한적 | **부적합** |
| 한국어 | 불가 | 제한적 | **부적합** |
| 추론 속도 | 빠름 (7.5s) | 보통 (11.8s) | **적합** |

**SmolVLM (256M, 500M)은 현재 건설현장 자재 손상 평가 용도로 부적합**

---

## 4. 권장 사항

### 4.1 단기 (프로토타입)

**Gemini 2.0 Flash API 사용 권장**
- 무료 (일 1500회)
- 고성능 비전 인식
- 한국어 지원
- JSON 출력 우수

```bash
# 실행 방법
set GEMINI_API_KEY=your_api_key
python comprehensive_vlm_benchmark.py gemini
```

### 4.2 중기 (개발)

1. **Qwen2.5-VL-7B** 또는 **LLaVA-1.6-7B** 로컬 배포
2. 건설 자재 이미지로 **Fine-tuning** 고려
3. 실제 손상 이미지 데이터셋 구축

### 4.3 장기 (배포)

| 시나리오 | 권장 모델 | 이유 |
|---------|---------|------|
| 모바일 앱 | Fine-tuned SmolVLM-2B | 온디바이스, 프라이버시 |
| 웹 서비스 | Gemini API | 비용 효율 |
| 오프라인 | Qwen2.5-VL-7B | 성능/크기 균형 |
| 최고 정확도 | GPT-4o | 비용 무관시 |

---

## 5. 다음 단계

### 5.1 Gemini API 테스트 실행
```bash
# 1. API 키 발급 (무료)
# https://aistudio.google.com/apikey

# 2. 환경변수 설정
set GEMINI_API_KEY=your_api_key_here

# 3. 테스트 실행
python comprehensive_vlm_benchmark.py gemini
```

### 5.2 실제 이미지 테스트
- 건설현장에서 촬영한 실제 안전모/안전화 이미지 수집
- 손상 유형별 레이블링
- 정확도 검증

### 5.3 Fine-tuning 검토
- LoRA 또는 QLoRA 방식
- 100-1000장의 건설 자재 이미지로 학습
- SmolVLM-2B 또는 Qwen2-VL-2B 대상

---

## 6. 파일 구조

```
P03_VLM_ondevice/
├── comprehensive_vlm_benchmark.py  # 종합 벤치마크 (모든 모델)
├── run_smolvlm_benchmark.py        # SmolVLM 전용 테스트
├── test_image_download.py          # 테스트 이미지 생성
├── VLM_BENCHMARK_GUIDE.md          # 테스트 기준 및 모델 가이드
├── VLM_BENCHMARK_RESULTS.md        # 이 보고서
├── smolvlm_benchmark_results.json  # SmolVLM 테스트 결과 (JSON)
└── test_images/                    # 테스트 이미지 폴더
    ├── synthetic_hardhat.jpg
    ├── synthetic_hardhat_cracked.jpg
    ├── synthetic_boots.jpg
    ├── synthetic_gloves.jpg
    ├── synthetic_hammer.jpg
    └── synthetic_hammer_rusty.jpg
```

---

## 7. 참고 자료

- [SmolVLM Blog - HuggingFace](https://huggingface.co/blog/smolvlm)
- [Top VLMs 2026 - DataCamp](https://www.datacamp.com/blog/top-vision-language-models)
- [Local VLMs - Roboflow](https://blog.roboflow.com/local-vision-language-models/)
- [Google AI Studio](https://aistudio.google.com/)
- [OSHA Hard Hat Safety](https://www.osha.gov/)

---

**작성**: Claude Code
**날짜**: 2026-02-26
