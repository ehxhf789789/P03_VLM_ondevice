use image::DynamicImage;
use ndarray::Array4;

/// 이미지 크기 상수 (SmolVLM-256M은 512x512 사용)
pub const IMAGE_SIZE: u32 = 512;

/// SigLIP 정규화 파라미터
const NORMALIZE_MEAN: f32 = 0.5;
const NORMALIZE_STD: f32 = 0.5;

/// 이미지를 바이트에서 로드한다.
pub fn load_image_from_bytes(bytes: &[u8]) -> Result<DynamicImage, String> {
    image::load_from_memory(bytes).map_err(|e| format!("Failed to load image: {}", e))
}

/// 이미지를 VLM 입력용 텐서로 전처리한다.
/// - 지정 크기로 리사이즈 (aspect ratio 유지, 패딩 없음)
/// - RGB [0,1] → SigLIP 정규화 (mean=0.5, std=0.5)
/// - shape: [1, 3, H, W] (NCHW)
pub fn preprocess_image(img: &DynamicImage) -> Array4<f32> {
    let resized = img.resize_exact(IMAGE_SIZE, IMAGE_SIZE, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();

    let mut tensor = Array4::<f32>::zeros((1, 3, IMAGE_SIZE as usize, IMAGE_SIZE as usize));

    for y in 0..IMAGE_SIZE as usize {
        for x in 0..IMAGE_SIZE as usize {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                let normalized = (val - NORMALIZE_MEAN) / NORMALIZE_STD;
                tensor[[0, c, y, x]] = normalized;
            }
        }
    }

    tensor
}

/// base64 인코딩된 이미지를 디코딩하여 전처리한다.
pub fn preprocess_from_base64(base64_str: &str) -> Result<Array4<f32>, String> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(base64_str)
        .map_err(|e| format!("Base64 decode error: {}", e))?;
    let img = load_image_from_bytes(&bytes)?;
    Ok(preprocess_image(&img))
}

/// 이미지 파일 경로에서 로드하여 전처리한다.
pub fn preprocess_from_path(path: &str) -> Result<(Array4<f32>, DynamicImage), String> {
    let img = image::open(path).map_err(|e| format!("Failed to open image: {}", e))?;
    let tensor = preprocess_image(&img);
    Ok((tensor, img))
}

/// 이미지를 base64로 인코딩하여 반환한다 (프론트엔드 미리보기용).
pub fn image_to_base64_thumbnail(img: &DynamicImage, max_size: u32) -> String {
    use base64::Engine;
    let thumb = img.thumbnail(max_size, max_size);
    let mut buf = std::io::Cursor::new(Vec::new());
    thumb
        .write_to(&mut buf, image::ImageFormat::Jpeg)
        .unwrap_or_default();
    let encoded = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());
    format!("data:image/jpeg;base64,{}", encoded)
}
