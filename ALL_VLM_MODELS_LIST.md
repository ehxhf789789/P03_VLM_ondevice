# 현존하는 VLM 모델 전체 목록 (2026년 2월 기준)

## 1. 로컬 실행 가능 모델 (오픈소스)

### 1.1 Tiny 모델 (< 1B) - 모바일/엣지 배포 가능

| 모델 | 크기 | VRAM | 제작사 | 특징 |
|------|-----|------|-------|------|
| **SmolVLM-256M** | 256M | ~500MB | HuggingFace | 최소형, 빠른 추론 |
| **SmolVLM-500M** | 500M | ~800MB | HuggingFace | 경량 VLM |
| **Florence-2-base** | 230M | ~1GB | Microsoft | 다양한 비전 태스크 |
| **Florence-2-large** | 770M | ~2GB | Microsoft | Caption, Detection, OCR |
| **MiniCPM-V-2** | 800M | ~2GB | OpenBMB | 모바일 최적화 |

### 1.2 Small 모델 (1-3B) - 일반 PC 실행 가능

| 모델 | 크기 | VRAM | 제작사 | 특징 |
|------|-----|------|-------|------|
| **Moondream2** | 1.6B | ~2GB | vikhyatk | 빠른 추론, 활발한 업데이트 |
| **SmolVLM-2B** | 2B | ~3GB | HuggingFace | 범용 VLM |
| **Qwen2-VL-2B** | 2B | ~4GB | Alibaba | 다국어, OCR 강점 |
| **PaliGemma-3B** | 3B | ~6GB | Google | 다국어, Fine-tuning 용이 |
| **DeepSeek-VL-1.3B** | 1.3B | ~3GB | DeepSeek | 추론 능력 우수 |
| **Phi-3-Vision-128k** | 4.2B | ~8GB | Microsoft | 긴 컨텍스트 |

### 1.3 Medium 모델 (7-13B) - GPU 서버 권장

| 모델 | 크기 | VRAM | 제작사 | 특징 |
|------|-----|------|-------|------|
| **Qwen2.5-VL-7B** | 7B | ~14GB | Alibaba | 최신, 고성능 |
| **LLaVA-1.6-7B** | 7B | ~14GB | 커뮤니티 | 범용, 문서화 우수 |
| **LLaVA-NeXT-7B** | 7B | ~14GB | 커뮤니티 | LLaVA 개선판 |
| **InternVL2-8B** | 8B | ~16GB | OpenGVLab | 벤치마크 상위 |
| **Idefics2-8B** | 8B | ~16GB | HuggingFace | 다국어 |
| **CogVLM2-8B** | 8B | ~16GB | Tsinghua | 고해상도 지원 |
| **MiniCPM-V-2.6** | 8B | ~16GB | OpenBMB | 동영상 지원 |
| **Llama-3.2-11B-Vision** | 11B | ~22GB | Meta | Llama 기반 |

### 1.4 Large 모델 (>20B) - 고성능 서버 필요

| 모델 | 크기 | VRAM | 제작사 | 특징 |
|------|-----|------|-------|------|
| **Qwen2.5-VL-72B** | 72B | ~140GB | Alibaba | 최고 성능 오픈소스 |
| **InternVL2-76B** | 76B | ~150GB | OpenGVLab | GPT-4V급 |
| **LLaVA-NeXT-72B** | 72B | ~140GB | 커뮤니티 | 대규모 LLaVA |
| **Llama-3.2-90B-Vision** | 90B | ~180GB | Meta | Llama 최대 |

---

## 2. 클라우드 API 모델

### 2.1 무료 티어 제공

| 모델 | 제공사 | 무료 한도 | API |
|------|-------|----------|-----|
| **Gemini 2.0 Flash** | Google | 15 req/min, 1500/day | `gemini-2.0-flash-exp` |
| **Gemini 1.5 Flash** | Google | 15 req/min | `gemini-1.5-flash` |
| **Gemini 1.5 Pro** | Google | 2 req/min | `gemini-1.5-pro` |
| **Groq LLaVA** | Groq | 제한적 | `llava-v1.5-7b-4096-preview` |

### 2.2 유료 API

| 모델 | 제공사 | 가격 (1M input tokens) | 특징 |
|------|-------|----------------------|------|
| **GPT-4o** | OpenAI | $5.00 | 최고 성능 |
| **GPT-4o-mini** | OpenAI | $0.15 | 가성비 |
| **GPT-4-Turbo-Vision** | OpenAI | $10.00 | 고해상도 |
| **Claude 3.5 Sonnet** | Anthropic | $3.00 | 코드 분석 강점 |
| **Claude 3 Opus** | Anthropic | $15.00 | 최고 추론 |
| **Claude 3 Haiku** | Anthropic | $0.25 | 빠른 응답 |
| **Gemini 1.5 Pro** | Google | $3.50 | 긴 컨텍스트 |

---

## 3. 벤치마크 성능 비교 (2026년 기준)

### 3.1 주요 벤치마크 점수

| 모델 | MMMU | DocVQA | ChartQA | TextVQA | 비고 |
|------|------|--------|---------|---------|------|
| GPT-4o | 69.1 | 92.8 | 85.7 | 77.4 | 최고 성능 |
| Claude 3.5 Sonnet | 68.3 | 95.2 | 90.8 | 74.1 | 문서 분석 최고 |
| Gemini 1.5 Pro | 62.2 | 93.1 | 87.2 | 78.7 | 균형 |
| Qwen2.5-VL-72B | 64.5 | 96.4 | 88.3 | 84.9 | 오픈소스 최고 |
| InternVL2-76B | 65.1 | 94.1 | 88.4 | 82.5 | 오픈소스 |
| Qwen2.5-VL-7B | 54.1 | 94.5 | 83.0 | 79.7 | 경량 고성능 |
| LLaVA-NeXT-7B | 44.1 | 87.5 | 69.6 | 64.9 | 범용 |
| SmolVLM-2B | 38.8 | 81.6 | 62.4 | 72.7 | 경량 |
| Florence-2 | N/A | 89.9 | 81.9 | 79.1 | 특화 태스크 |

### 3.2 손상 탐지/객체 인식 관련 성능 (예상)

| 모델 | 객체 인식 | 손상 탐지 | JSON 출력 | 한국어 | 권장도 |
|------|---------|---------|----------|-------|-------|
| GPT-4o | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | 최고 |
| Claude 3.5 | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★☆ | 높음 |
| Gemini 1.5 Pro | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ | 높음 |
| Qwen2.5-VL-7B | ★★★★☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ | 권장 |
| LLaVA-1.6-7B | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ | 보통 |
| Florence-2 | ★★★★☆ | ★★☆☆☆ | ★★☆☆☆ | ★☆☆☆☆ | 제한적 |
| SmolVLM-2B | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★☆☆☆ | 제한적 |
| SmolVLM-500M | ★☆☆☆☆ | ★☆☆☆☆ | ★★☆☆☆ | ★☆☆☆☆ | 부적합 |

---

## 4. 건설현장 자재 손상 평가 권장 모델

### 4.1 프로토타입/개발 단계

| 순위 | 모델 | 이유 |
|-----|------|------|
| 1 | **Gemini 2.0 Flash** | 무료, 고성능, 한국어 지원 |
| 2 | **Qwen2.5-VL-7B** | 오픈소스, 한국어 우수, 로컬 가능 |
| 3 | **GPT-4o-mini** | 저비용, 고성능 |

### 4.2 온디바이스 배포

| 순위 | 모델 | 이유 |
|-----|------|------|
| 1 | **Fine-tuned Qwen2-VL-2B** | 크기/성능 균형 |
| 2 | **Fine-tuned SmolVLM-2B** | 경량, 빠름 |
| 3 | **Florence-2 + Custom Head** | 특화 태스크 |

### 4.3 서버 배포 (정확도 중시)

| 순위 | 모델 | 이유 |
|-----|------|------|
| 1 | **Qwen2.5-VL-72B** | 오픈소스 최고 |
| 2 | **InternVL2-8B** | 가성비 |
| 3 | **LLaVA-NeXT-13B** | 범용 |

---

## 5. 설치 및 실행 예시

### 5.1 SmolVLM
```python
from transformers import AutoProcessor, AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
```

### 5.2 Qwen2-VL
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
```

### 5.3 Florence-2
```python
from transformers import AutoProcessor, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
```

### 5.4 Gemini API (무료)
```python
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-2.0-flash-exp")
response = model.generate_content([prompt, image])
```

---

## 참고 자료

- [HuggingFace VLM Models](https://huggingface.co/models?pipeline_tag=image-text-to-text)
- [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)
- [Google AI Studio](https://aistudio.google.com/)
- [SmolVLM Blog](https://huggingface.co/blog/smolvlm)
- [Qwen2-VL GitHub](https://github.com/QwenLM/Qwen2-VL)
