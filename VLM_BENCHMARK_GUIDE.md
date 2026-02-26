# VLM 성능 벤치마크 - 건설현장 자재 손상 평가

## 1. 프로젝트 개요

### 목표
건설현장 자재(안전모, 안전화, 장갑, 공구 등)의 손상을 VLM(Vision-Language Model)로 자동 인식하여:
- 객체 인식 (Object Detection)
- 손상 위치 파악 (Damage Localization)
- 손상 정도 평가 (Severity Assessment)
- 정형 데이터 출력 (Structured Output)

### 온디바이스 요구사항
- 무료 또는 저비용 솔루션
- 모바일/엣지 디바이스 실행 가능
- 프라이버시 보호 (로컬 처리)

---

## 2. 테스트 이미지 기준

### 2.1 자재 카테고리

| 카테고리 | 품목 | 손상 유형 |
|---------|------|----------|
| **안전모 (Hard Hat)** | 건설용 헬멧, ABS 헬멧 | 균열, 스크래치, 변색, UV 손상, 찌그러짐 |
| **안전화 (Safety Boots)** | 강철토캡, 작업화 | 마모, 찢어짐, 밑창 분리, 토캡 노출 |
| **작업장갑 (Work Gloves)** | 가죽장갑, 고무장갑 | 마모, 구멍, 찢어짐, 코팅 벗겨짐 |
| **공구 (Tools)** | 몽키스패너, 해머, 드라이버 | 녹, 손잡이 손상, 날 마모 |

### 2.2 테스트 이미지 목록

#### 정상 상태 이미지 (기준선)
| ID | 설명 | 출처 |
|----|------|------|
| `hardhat_yellow_new` | 노란색 안전모 (신품) | Wikimedia Commons |
| `hardhat_orange` | 주황색 안전모 | Wikimedia Commons |
| `safety_boots_steel_toe` | 강철 토캡 안전화 | Wikimedia Commons |
| `work_gloves_leather` | 가죽 작업 장갑 | Wikimedia Commons |
| `wrench_adjustable` | 몽키스패너 | Wikimedia Commons |
| `hammer_claw` | 장도리 | Wikimedia Commons |

### 2.3 손상 평가 기준

| 등급 | 상태 | 조치 |
|-----|------|------|
| **New** | 신품, 사용 흔적 없음 | 정상 사용 |
| **Good** | 경미한 사용 흔적 | 정상 사용 |
| **Fair** | 육안 확인 가능한 마모 | 모니터링 필요 |
| **Poor** | 상당한 손상 | 점검 필요 |
| **Damaged** | 심각한 손상 | 즉시 교체 |

---

## 3. VLM 모델 현황 (2026년 2월 기준)

### 3.1 로컬 실행 모델 (온디바이스)

#### Tiny 모델 (< 1B)
| 모델 | 크기 | GPU 메모리 | 특징 |
|------|-----|----------|------|
| **SmolVLM-256M** | 256M | ~500MB | 최소형, 모바일 최적화 |
| **SmolVLM-500M** | 500M | ~800MB | 균형형 |
| **Moondream2** | 1.6B | ~2GB | 빠른 추론 |

#### Small 모델 (1-3B)
| 모델 | 크기 | GPU 메모리 | 특징 |
|------|-----|----------|------|
| **SmolVLM-2B** | 2B | ~3GB | HuggingFace 공식 |
| **PaliGemma-3B** | 3B | ~4GB | Google, 다국어 |
| **Qwen2-VL-2B** | 2B | ~3GB | Alibaba, OCR 강점 |
| **DeepSeek-VL-1.3B** | 1.3B | ~2GB | 추론 능력 우수 |

#### Medium 모델 (7-8B)
| 모델 | 크기 | GPU 메모리 | 특징 |
|------|-----|----------|------|
| **Qwen2.5-VL-7B** | 7B | ~10GB | 최신, 고성능 |
| **LLaVA-1.6-7B** | 7B | ~10GB | 범용, 커뮤니티 활성 |
| **InternVL2-8B** | 8B | ~12GB | 벤치마크 상위권 |
| **Gemma3-4B** | 4B | ~6GB | Google, 경량 |

### 3.2 클라우드 API 모델

#### 무료 티어
| 모델 | 제공사 | 무료 한도 | 비고 |
|------|-------|----------|------|
| **Gemini 2.0 Flash** | Google | 15 req/min, 1500/day | 추천 |
| **Gemini 1.5 Pro** | Google | 2 req/min | 고성능 |
| **Groq LLaVA** | Groq | 제한적 | 빠른 추론 |

#### 유료 (저비용)
| 모델 | 제공사 | 가격 (1M tokens) | 비고 |
|------|-------|-----------------|------|
| **GPT-4o-mini** | OpenAI | $0.15 input | 가성비 |
| **Claude 3 Haiku** | Anthropic | $0.25 input | 빠름 |

#### 유료 (고성능)
| 모델 | 제공사 | 가격 (1M tokens) | 비고 |
|------|-------|-----------------|------|
| **GPT-4o** | OpenAI | $5.00 input | 최고 성능 |
| **Claude 3.5 Sonnet** | Anthropic | $3.00 input | 코드 강점 |
| **Gemini 1.5 Pro** | Google | $3.50 input | 긴 컨텍스트 |

### 3.3 성능 비교 요약

```
┌─────────────────────────────────────────────────────────────┐
│                    성능 vs 비용 매트릭스                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  성능 ↑                                                     │
│     │    ★ GPT-4o              ★ Claude 3.5 Sonnet         │
│     │         ★ Gemini 1.5 Pro                             │
│     │                                                       │
│     │         ★ Qwen2.5-VL-7B    ★ InternVL2-8B            │
│     │     ★ GPT-4o-mini                                    │
│     │                                                       │
│     │  ★ Gemini 2.0 Flash (FREE!)                          │
│     │         ★ LLaVA-1.6-7B                               │
│     │                                                       │
│     │     ★ SmolVLM-2B          ★ Qwen2-VL-2B              │
│     │                                                       │
│     │  ★ SmolVLM-500M                                      │
│     │  ★ SmolVLM-256M                                      │
│     │                                                       │
│     └───────────────────────────────────────────────→ 비용  │
│         무료        저비용        중간        고비용          │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 평가 프롬프트

### 4.1 객체 인식 (Object Recognition)
```
이 이미지에서 보이는 물체를 식별하세요.
답변 형식:
- 물체 이름: [한국어로]
- 유형: [안전모/안전화/장갑/공구 중 선택]
- 색상: [관찰된 색상]
```

### 4.2 손상 탐지 (Damage Detection)
```
이 건설현장 자재/장비 이미지를 분석하여 손상 여부를 평가하세요.

다음 항목을 확인하고 보고하세요:
1. 물체 종류
2. 손상 유무 (있음/없음)
3. 손상 유형 (균열, 스크래치, 변색, 마모, 녹, 찢어짐 등)
4. 손상 위치 (상단/중앙/하단, 좌측/우측/전면/후면)
5. 손상 심각도 (경미/보통/심각)
6. 사용 가능 여부 (가능/점검필요/교체필요)
```

### 4.3 정형 JSON 출력
```json
{
    "object": {
        "name": "물체 이름",
        "type": "안전모|안전화|장갑|공구",
        "color": "색상"
    },
    "condition": {
        "overall": "new|good|fair|poor|damaged",
        "usable": true/false,
        "recommendation": "계속사용|점검필요|교체권장|즉시교체"
    },
    "damages": [
        {
            "type": "손상유형",
            "location": "위치설명",
            "severity": "minor|moderate|severe",
            "size_percent": 0-100
        }
    ]
}
```

---

## 5. 실행 방법

### 5.1 환경 설정
```bash
# 필수 패키지
pip install torch transformers pillow requests

# Gemini API (무료)
pip install google-generativeai
set GEMINI_API_KEY=your_api_key

# OpenAI API (유료)
pip install openai
set OPENAI_API_KEY=your_api_key

# Anthropic API (유료)
pip install anthropic
set ANTHROPIC_API_KEY=your_api_key
```

### 5.2 벤치마크 실행
```bash
# Gemini 빠른 테스트 (추천, 무료)
python comprehensive_vlm_benchmark.py gemini

# 로컬 모델 테스트
python comprehensive_vlm_benchmark.py local

# 전체 벤치마크
python comprehensive_vlm_benchmark.py full
```

---

## 6. 권장 사항

### 온디바이스 배포
1. **프로토타입**: Gemini 2.0 Flash (무료 API)로 성능 확인
2. **개발**: SmolVLM-2B 또는 Qwen2-VL-2B로 로컬 개발
3. **배포**: 모바일은 SmolVLM-500M, 서버는 Qwen2.5-VL-7B

### 비용 최적화
- 무료: Gemini 2.0 Flash (일 1500회)
- 저비용: GPT-4o-mini ($0.15/1M tokens)
- 로컬: SmolVLM (GPU 필요)

---

## 참고 자료

- [SmolVLM - HuggingFace](https://huggingface.co/blog/smolvlm)
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)
- [Google AI Studio](https://aistudio.google.com/)
- [Top VLMs 2026 - DataCamp](https://www.datacamp.com/blog/top-vision-language-models)
- [Local VLMs - Roboflow](https://blog.roboflow.com/local-vision-language-models/)
