"""
Google Gemini API 테스트 (무료 티어)
API 키 필요: https://aistudio.google.com/apikey
"""
import os
import base64
import json
import numpy as np
from PIL import Image
import io

# API 키 설정 방법 안내
API_KEY = os.environ.get("GEMINI_API_KEY", "")

if not API_KEY:
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Google Gemini API 키가 필요합니다 (무료)                    ║
╠══════════════════════════════════════════════════════════════╣
║  1. https://aistudio.google.com/apikey 접속                  ║
║  2. Google 계정으로 로그인                                   ║
║  3. "Create API Key" 클릭                                    ║
║  4. 생성된 키 복사                                           ║
║                                                              ║
║  실행 방법:                                                  ║
║  set GEMINI_API_KEY=your_api_key_here                        ║
║  python test_gemini_free.py                                  ║
╚══════════════════════════════════════════════════════════════╝

무료 한도: 분당 15회, 일 1500회 (충분함!)
""")
    exit(1)

try:
    import google.generativeai as genai
except ImportError:
    print("Installing google-generativeai...")
    os.system("pip install google-generativeai -q")
    import google.generativeai as genai

# API 설정
genai.configure(api_key=API_KEY)

def create_test_image():
    """손상된 표면 테스트 이미지 생성"""
    img = np.ones((384, 384, 3), dtype=np.uint8) * 170
    # 녹 (우측 상단)
    img[50:130, 250:350] = [140, 85, 40]
    # 스크래치 (대각선)
    for i in range(100, 280):
        img[i, i-30:i-27] = [50, 45, 40]
    # 찌그러짐 (좌측 하단)
    for y in range(260, 340):
        for x in range(60, 140):
            dist = np.sqrt((x-100)**2 + (y-300)**2)
            if dist < 40:
                img[y, x] = [int(170 - (40-dist)*1.5)] * 3
    return Image.fromarray(img)

def test_gemini(image, prompt, test_name):
    """Gemini API 테스트"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")

    model = genai.GenerativeModel("gemini-2.0-flash")

    response = model.generate_content([prompt, image])

    print(f"Response:\n{response.text}")
    return response.text

def main():
    print("Google Gemini Vision Test (FREE API)")
    print("="*60)

    # 테스트 이미지 생성
    image = create_test_image()
    image.save("test_gemini_image.png")
    print("Test image saved!")

    # 테스트 1: 기본 인식
    test_gemini(image,
        "What do you see in this image? Describe the surface and any visible marks.",
        "Basic Recognition"
    )

    # 테스트 2: 손상 분석
    test_gemini(image,
        """Analyze this surface image for damage.
Identify:
1. Type of damage (rust, scratch, dent, etc.)
2. Location (top-left, center, bottom-right, etc.)
3. Severity (minor/moderate/severe)""",
        "Damage Analysis"
    )

    # 테스트 3: JSON 출력
    response = test_gemini(image,
        """Analyze this image for damage and output ONLY valid JSON:
{
  "object_type": "surface type",
  "condition": "good/fair/poor/damaged",
  "damages": [
    {
      "type": "damage type",
      "location": "position description",
      "severity": "minor/moderate/severe",
      "size_percent": estimated percentage
    }
  ]
}""",
        "Structured JSON Output"
    )

    # JSON 파싱 테스트
    print("\n--- JSON Parsing Test ---")
    try:
        # JSON 블록 추출
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "{" in response:
            json_str = response[response.find("{"):response.rfind("}")+1]
        else:
            json_str = response

        parsed = json.loads(json_str)
        print(f"SUCCESS: Valid JSON!")
        print(json.dumps(parsed, indent=2))
    except Exception as e:
        print(f"Parse error: {e}")

    # 테스트 4: 위치 좌표
    test_gemini(image,
        """Detect all damage in this image.
For each damage, provide approximate bounding box as percentage:
- x_min: left edge (0-100%)
- y_min: top edge (0-100%)
- x_max: right edge (0-100%)
- y_max: bottom edge (0-100%)

Output JSON array.""",
        "Bounding Box Estimation"
    )

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
