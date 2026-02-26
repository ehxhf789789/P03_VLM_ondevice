"""
테스트 이미지 다운로드 및 확인
SSL 인증서 문제 우회
"""
import requests
from PIL import Image
from io import BytesIO
import os
import urllib3
import warnings

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

# 다양한 소스에서 이미지 URL
TEST_IMAGES = {
    # Wikimedia Commons (SSL verify=False로 시도)
    "hardhat_yellow": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Bauhelm_gelb.jpg/640px-Bauhelm_gelb.jpg",
    "hardhat_white": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Hard_hat.jpg/640px-Hard_hat.jpg",
    "safety_boots": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Sicherheitsschuhe.jpg/640px-Sicherheitsschuhe.jpg",
    "work_gloves": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Leather_gloves.JPG/640px-Leather_gloves.JPG",
    "hammer": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Claw-hammer.jpg/640px-Claw-hammer.jpg",
    "screwdriver": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Screw_Driver.jpg/640px-Screw_Driver.jpg",
}

def download_image(name, url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        # SSL 검증 비활성화
        response = requests.get(url, headers=headers, timeout=30, verify=False)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 저장
        os.makedirs("test_images", exist_ok=True)
        filename = f"test_images/{name}.jpg"
        img.save(filename, "JPEG", quality=90)

        print(f"OK: {name} ({img.size[0]}x{img.size[1]}) -> {filename}")
        return img
    except Exception as e:
        print(f"FAIL: {name} - {str(e)[:80]}")
        return None

def create_synthetic_images():
    """외부 다운로드 실패시 합성 이미지 생성"""
    import numpy as np

    os.makedirs("test_images", exist_ok=True)
    images = {}

    # 안전모 시뮬레이션 (노란색 반원형)
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240  # 밝은 배경
    # 노란색 헬멧 형태
    for y in range(100, 300):
        for x in range(100, 300):
            if (x-200)**2 + (y-200)**2 < 90**2:  # 원형 영역
                if y < 200:  # 상단만 노란색
                    img[y, x] = [0, 200, 255]  # BGR -> Yellow
                else:
                    img[y, x] = [0, 180, 230]  # 약간 어두운 노란색
    # 챙 부분
    img[195:210, 80:320] = [50, 50, 60]

    hardhat = Image.fromarray(img)
    hardhat.save("test_images/synthetic_hardhat.jpg")
    images["hardhat"] = hardhat
    print("Created: synthetic_hardhat.jpg (노란색 안전모)")

    # 손상된 안전모 (균열 추가)
    damaged = img.copy()
    # 균열 시뮬레이션
    for i in range(120, 200):
        damaged[i, 180+i//10:185+i//10] = [30, 30, 40]
    damaged_img = Image.fromarray(damaged)
    damaged_img.save("test_images/synthetic_hardhat_cracked.jpg")
    images["hardhat_cracked"] = damaged_img
    print("Created: synthetic_hardhat_cracked.jpg (균열 있는 안전모)")

    # 안전화 시뮬레이션
    boot = np.ones((400, 400, 3), dtype=np.uint8) * 230
    # 부츠 형태 (검은색)
    boot[150:350, 100:300] = [40, 30, 20]  # 본체
    boot[320:360, 80:320] = [30, 20, 15]   # 밑창
    boot[150:200, 250:310] = [60, 50, 40]  # 발목 부분 밝게

    boot_img = Image.fromarray(boot)
    boot_img.save("test_images/synthetic_boots.jpg")
    images["boots"] = boot_img
    print("Created: synthetic_boots.jpg (안전화)")

    # 장갑 시뮬레이션
    glove = np.ones((400, 400, 3), dtype=np.uint8) * 220
    # 장갑 형태 (갈색)
    glove[100:350, 120:280] = [60, 100, 150]  # 손바닥
    # 손가락들
    for i, x in enumerate([130, 160, 190, 220, 250]):
        glove[50:120, x:x+25] = [70, 110, 160]

    glove_img = Image.fromarray(glove)
    glove_img.save("test_images/synthetic_gloves.jpg")
    images["gloves"] = glove_img
    print("Created: synthetic_gloves.jpg (작업장갑)")

    # 공구 - 해머 시뮬레이션
    hammer = np.ones((400, 400, 3), dtype=np.uint8) * 235
    # 해머 헤드 (회색)
    hammer[100:180, 150:280] = [80, 80, 90]
    # 손잡이 (갈색)
    hammer[180:350, 195:220] = [50, 90, 130]

    hammer_img = Image.fromarray(hammer)
    hammer_img.save("test_images/synthetic_hammer.jpg")
    images["hammer"] = hammer_img
    print("Created: synthetic_hammer.jpg (해머)")

    # 녹슨 해머
    rusty_hammer = hammer.copy()
    rusty_hammer[110:170, 160:270] = [45, 75, 140]  # 녹 색상
    rusty_img = Image.fromarray(rusty_hammer)
    rusty_img.save("test_images/synthetic_hammer_rusty.jpg")
    images["hammer_rusty"] = rusty_img
    print("Created: synthetic_hammer_rusty.jpg (녹슨 해머)")

    return images

if __name__ == "__main__":
    print("=" * 60)
    print("테스트 이미지 다운로드")
    print("=" * 60)

    downloaded = {}
    for name, url in TEST_IMAGES.items():
        img = download_image(name, url)
        if img:
            downloaded[name] = img

    print(f"\n외부 다운로드: {len(downloaded)}/{len(TEST_IMAGES)} 성공")

    if len(downloaded) < 3:
        print("\n외부 다운로드 실패. 합성 이미지 생성 중...")
        print("-" * 60)
        synthetic = create_synthetic_images()
        print(f"\n합성 이미지 {len(synthetic)}개 생성 완료")

    print("\n" + "=" * 60)
    print("test_images/ 폴더에 이미지가 저장되었습니다.")
