"""
모든 주요 로컬 VLM 모델 벤치마크
파인튜닝 없이 Zero-shot 성능 평가
"""
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'

import torch
from PIL import Image

# ============================================================================
# 테스트할 VLM 모델 전체 목록
# ============================================================================

ALL_VLM_MODELS = {
    # === Tiny (< 1B) ===
    "SmolVLM-256M": {
        "id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "type": "smolvlm",
        "size": "256M",
        "vram": "~500MB"
    },
    "SmolVLM-500M": {
        "id": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "type": "smolvlm",
        "size": "500M",
        "vram": "~800MB"
    },

    # === Small (1-3B) ===
    "Moondream2": {
        "id": "vikhyatk/moondream2",
        "type": "moondream",
        "size": "1.6B",
        "vram": "~2GB"
    },
    "SmolVLM-2B": {
        "id": "HuggingFaceTB/SmolVLM-Instruct",
        "type": "smolvlm",
        "size": "2B",
        "vram": "~3GB"
    },
    "Qwen2-VL-2B": {
        "id": "Qwen/Qwen2-VL-2B-Instruct",
        "type": "qwen2vl",
        "size": "2B",
        "vram": "~4GB"
    },
    "PaliGemma-3B": {
        "id": "google/paligemma-3b-pt-224",
        "type": "paligemma",
        "size": "3B",
        "vram": "~6GB"
    },
    "Florence-2-base": {
        "id": "microsoft/Florence-2-base",
        "type": "florence",
        "size": "0.2B",
        "vram": "~1GB"
    },
    "Florence-2-large": {
        "id": "microsoft/Florence-2-large",
        "type": "florence",
        "size": "0.7B",
        "vram": "~2GB"
    },

    # === Medium (4-8B) ===
    "Phi-3.5-Vision": {
        "id": "microsoft/Phi-3.5-vision-instruct",
        "type": "phi3v",
        "size": "4.2B",
        "vram": "~8GB"
    },
    "LLaVA-1.5-7B": {
        "id": "llava-hf/llava-1.5-7b-hf",
        "type": "llava",
        "size": "7B",
        "vram": "~14GB"
    },
    "Qwen2-VL-7B": {
        "id": "Qwen/Qwen2-VL-7B-Instruct",
        "type": "qwen2vl",
        "size": "7B",
        "vram": "~14GB"
    },
    "InternVL2-8B": {
        "id": "OpenGVLab/InternVL2-8B",
        "type": "internvl",
        "size": "8B",
        "vram": "~16GB"
    },
}

# ============================================================================
# 프롬프트
# ============================================================================

TEST_PROMPT = """Analyze this image of construction safety equipment.

1. What object is shown? (hardhat, safety boots, gloves, hammer, wrench, etc.)
2. What color is it?
3. Is there any damage? (yes/no)
4. If damaged, describe: type, location, severity

Output as JSON:
{
    "object": "name",
    "color": "color",
    "damaged": true/false,
    "damage_details": {"type": "", "location": "", "severity": ""}
}"""

# ============================================================================
# 모델별 추론 함수
# ============================================================================

def infer_smolvlm(model_id, image, prompt, device):
    """SmolVLM 추론"""
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)

    response = processor.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()

    del model, processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return response

def infer_qwen2vl(model_id, image, prompt, device):
    """Qwen2-VL 추론"""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300)

    response = processor.decode(outputs[0], skip_special_tokens=True)

    del model, processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return response

def infer_florence(model_id, image, prompt, device):
    """Florence-2 추론"""
    from transformers import AutoProcessor, AutoModelForCausalLM

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # Florence는 특정 태스크 형식 사용
    task = "<DETAILED_CAPTION>"
    inputs = processor(text=task, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300)

    response = processor.decode(outputs[0], skip_special_tokens=True)

    del model, processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return response

def infer_llava(model_id, image, prompt, device):
    """LLaVA 추론"""
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300)

    response = processor.decode(outputs[0], skip_special_tokens=True)

    del model, processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return response

def infer_phi3v(model_id, image, prompt, device):
    """Phi-3-Vision 추론"""
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]
    text = processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text, [image], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300)

    response = processor.decode(outputs[0], skip_special_tokens=True)

    del model, processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return response

def infer_paligemma(model_id, image, prompt, device):
    """PaliGemma 추론"""
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300)

    response = processor.decode(outputs[0], skip_special_tokens=True)

    del model, processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return response

def infer_moondream(model_id, image, prompt, device):
    """Moondream 추론"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    enc_image = model.encode_image(image)
    response = model.answer_question(enc_image, prompt, tokenizer)

    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return response

# ============================================================================
# 테스트 실행
# ============================================================================

def test_model(model_name, model_info, images, device):
    """단일 모델 테스트"""
    print(f"\n{'='*70}")
    print(f"Model: {model_name} ({model_info['size']})")
    print(f"ID: {model_info['id']}")
    print(f"VRAM: {model_info['vram']}")
    print(f"{'='*70}")

    result = {
        "model": model_name,
        "model_id": model_info["id"],
        "size": model_info["size"],
        "type": model_info["type"],
        "tests": [],
        "summary": {}
    }

    # 추론 함수 선택
    infer_funcs = {
        "smolvlm": infer_smolvlm,
        "qwen2vl": infer_qwen2vl,
        "florence": infer_florence,
        "llava": infer_llava,
        "phi3v": infer_phi3v,
        "paligemma": infer_paligemma,
        "moondream": infer_moondream,
    }

    infer_fn = infer_funcs.get(model_info["type"])
    if not infer_fn:
        print(f"  Unknown model type: {model_info['type']}")
        result["error"] = f"Unknown type: {model_info['type']}"
        return result

    total_time = 0
    successful = 0
    scores = {"object": 0, "damage": 0, "json": 0}

    for img_name, img_data in images.items():
        print(f"\n  Image: {img_name}...", end=" ", flush=True)

        try:
            start = time.time()
            response = infer_fn(model_info["id"], img_data["image"], TEST_PROMPT, device)
            elapsed = time.time() - start
            total_time += elapsed
            successful += 1

            # 점수 계산
            resp_lower = response.lower()
            if any(w in resp_lower for w in ["hardhat", "helmet", "boot", "glove", "hammer", "tool"]):
                scores["object"] += 1
            if any(w in resp_lower for w in ["damage", "crack", "rust", "wear", "no damage", "good"]):
                scores["damage"] += 1
            if "{" in response and "}" in response:
                try:
                    json.loads(response[response.find("{"):response.rfind("}")+1])
                    scores["json"] += 1
                except:
                    pass

            result["tests"].append({
                "image": img_name,
                "response": response[:400],
                "time_sec": round(elapsed, 2),
                "success": True
            })

            print(f"OK ({elapsed:.1f}s)")
            print(f"      >> {response[:80]}...")

        except Exception as e:
            result["tests"].append({
                "image": img_name,
                "error": str(e)[:100],
                "success": False
            })
            print(f"FAIL: {str(e)[:50]}")

    # 요약
    result["summary"] = {
        "total": len(images),
        "successful": successful,
        "avg_time": round(total_time / successful, 2) if successful > 0 else 0,
        "total_time": round(total_time, 2)
    }
    result["scores"] = scores

    return result

def load_images():
    """테스트 이미지 로드"""
    images = {}
    for path in Path("test_images").glob("*.jpg"):
        try:
            images[path.stem] = {"image": Image.open(path).convert("RGB"), "path": str(path)}
        except:
            pass
    return images

def main(models_to_test=None):
    """메인"""
    print("=" * 70)
    print("VLM Zero-shot Performance Benchmark")
    print("Construction Equipment Damage Assessment")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    images = load_images()
    print(f"Images: {len(images)}")

    if models_to_test:
        models = {k: v for k, v in ALL_VLM_MODELS.items() if k in models_to_test}
    else:
        # 기본: 작은 모델들만
        models = {k: v for k, v in ALL_VLM_MODELS.items()
                  if v["size"] in ["256M", "500M", "0.2B", "0.7B", "1.6B", "2B"]}

    print(f"Models to test: {len(models)}")
    for name in models:
        print(f"  - {name}")

    all_results = []

    for name, info in models.items():
        result = test_model(name, info, images, device)
        all_results.append(result)

        # 중간 저장
        with open("all_vlm_results.json", "w", encoding="utf-8") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "results": all_results}, f, indent=2, ensure_ascii=False)

    # 요약
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in all_results:
        if "scores" in r and r["scores"]:
            total = sum(r["scores"].values())
            print(f"\n{r['model']} ({r['size']}):")
            print(f"  Score: {total}/{len(images)*3} ({total/(len(images)*3)*100:.0f}%)")
            print(f"  Time: {r['summary'].get('avg_time', 'N/A')}s avg")
        elif "error" in r:
            print(f"\n{r['model']}: FAILED - {r.get('error', 'Unknown')[:50]}")

    return all_results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        # 작은 모델들 테스트
        main(["SmolVLM-256M", "SmolVLM-500M", "Florence-2-base", "Florence-2-large"])
