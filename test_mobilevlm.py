"""MobileVLM 테스트 - 모바일 특화 VLM"""
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
from PIL import Image
import numpy as np

# 테스트할 경량 모델들
MODELS_TO_TEST = [
    ("vikhyatk/moondream2", "Moondream2 (1.6B)"),
    # ("mtgv/MobileVLM-1.7B", "MobileVLM 1.7B"),  # 다른 로딩 방식 필요
]

def create_test_image():
    """손상된 표면 테스트 이미지"""
    img = np.ones((384, 384, 3), dtype=np.uint8) * 170
    # 녹 (우측 상단)
    img[50:130, 250:350] = [140, 85, 40]
    # 스크래치 (대각선)
    for i in range(100, 280):
        img[i, i-30:i-27] = [50, 45, 40]
    return Image.fromarray(img)

def test_moondream():
    """Moondream2 테스트 (경량 VLM)"""
    print("="*60)
    print("Testing: Moondream2 (1.6B)")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM

        model_id = "vikhyatk/moondream2"
        revision = "2025-01-09"

        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=True,
            torch_dtype=torch.float32
        )
        print("Model loaded!")

        image = create_test_image()

        # 테스트 프롬프트들
        prompts = [
            "Describe what you see in this image.",
            "Is there any damage visible? Describe type and location.",
            "Output JSON: {\"object\": \"\", \"damage\": \"\", \"severity\": \"\"}"
        ]

        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            enc_image = model.encode_image(image)
            answer = model.answer_question(enc_image, prompt, tokenizer)
            print(f"Response: {answer}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Mobile VLM Benchmark")
    print("Testing lightweight models for smartphone deployment\n")

    # GPU 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cpu":
        print("WARNING: Running on CPU will be slow")

    test_moondream()

    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)
