use ort::session::Session;
use std::path::Path;

use super::preprocessing;
use super::tokenizer::TokenizerWrapper;

/// 최대 생성 토큰 수
const MAX_NEW_TOKENS: usize = 512;

/// 비전 특징 벡터 (shape dims + flat data)
struct Features {
    shape: Vec<i64>,
    data: Vec<f32>,
}

/// SmolVLM ONNX 3-세션 추론 엔진
pub struct SmolVlmEngine {
    vision_encoder: Session,
    embed_tokens: Session,
    decoder: Session,
    tokenizer: TokenizerWrapper,
}

impl SmolVlmEngine {
    /// 모델 디렉토리에서 세 개의 ONNX 세션을 로드한다.
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        log::info!("Loading ONNX sessions from {:?}", model_dir);

        let vision_encoder = Session::builder()
            .and_then(|b| b.commit_from_file(model_dir.join("vision_encoder.onnx")))
            .map_err(|e| format!("Failed to load vision_encoder: {}", e))?;

        let embed_tokens = Session::builder()
            .and_then(|b| b.commit_from_file(model_dir.join("embed_tokens.onnx")))
            .map_err(|e| format!("Failed to load embed_tokens: {}", e))?;

        let decoder = Session::builder()
            .and_then(|b| b.commit_from_file(model_dir.join("decoder_model_merged.onnx")))
            .map_err(|e| format!("Failed to load decoder: {}", e))?;

        let tokenizer = TokenizerWrapper::from_file(&model_dir.join("tokenizer.json"))?;

        log::info!("All ONNX sessions loaded successfully");
        Ok(Self {
            vision_encoder,
            embed_tokens,
            decoder,
            tokenizer,
        })
    }

    /// 이미지 바이트와 텍스트 프롬프트로 추론을 실행한다.
    pub fn run_inference(
        &mut self,
        image_bytes: &[u8],
        prompt: &str,
    ) -> Result<String, String> {
        let img = preprocessing::load_image_from_bytes(image_bytes)?;
        let pixel_values = preprocessing::preprocess_image(&img);

        let pv_shape: Vec<usize> = pixel_values.shape().to_vec();
        let (pv_data, _offset) = pixel_values.into_raw_vec_and_offset();
        let image_features = self.encode_vision(&pv_shape, &pv_data)?;

        let token_ids = self.tokenizer.encode(prompt)?;
        let output_ids = self.generate(&image_features, &token_ids)?;
        self.tokenizer.decode(&output_ids)
    }

    /// 이미지 파일 경로에서 추론을 실행한다.
    pub fn run_inference_from_path(
        &mut self,
        image_path: &str,
        prompt: &str,
    ) -> Result<String, String> {
        let (pixel_values, _img) = preprocessing::preprocess_from_path(image_path)?;
        let pv_shape: Vec<usize> = pixel_values.shape().to_vec();
        let (pv_data, _offset) = pixel_values.into_raw_vec_and_offset();
        let image_features = self.encode_vision(&pv_shape, &pv_data)?;
        let token_ids = self.tokenizer.encode(prompt)?;
        let output_ids = self.generate(&image_features, &token_ids)?;
        self.tokenizer.decode(&output_ids)
    }

    /// 비전 인코더를 실행한다.
    fn encode_vision(&mut self, pv_shape: &[usize], pv_data: &[f32]) -> Result<Features, String> {
        let input_value = ort::value::Value::from_array(
            (pv_shape.to_vec(), pv_data.to_vec()),
        )
        .map_err(|e| format!("Vision input error: {}", e))?;

        let outputs = self
            .vision_encoder
            .run(ort::inputs!["pixel_values" => input_value])
            .map_err(|e| format!("Vision encoder error: {}", e))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Vision output extract error: {}", e))?;

        // Shape derefs to [i64]
        Ok(Features {
            shape: shape.to_vec(),
            data: data.to_vec(),
        })
    }

    /// 텍스트 토큰을 임베딩한다.
    fn embed_text_tokens(&mut self, token_ids: &[i64]) -> Result<Features, String> {
        let input_value = ort::value::Value::from_array(
            (vec![1, token_ids.len()], token_ids.to_vec()),
        )
        .map_err(|e| format!("Embed input error: {}", e))?;

        let outputs = self
            .embed_tokens
            .run(ort::inputs!["input_ids" => input_value])
            .map_err(|e| format!("Embed tokens error: {}", e))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Embed output extract error: {}", e))?;

        Ok(Features {
            shape: shape.to_vec(),
            data: data.to_vec(),
        })
    }

    /// 자기회귀 생성을 수행한다.
    fn generate(
        &mut self,
        _image_features: &Features,
        prompt_token_ids: &[i64],
    ) -> Result<Vec<i64>, String> {
        let eos_id = self.tokenizer.eos_token_id();
        let mut generated_ids: Vec<i64> = Vec::new();
        let mut current_ids = prompt_token_ids.to_vec();

        for _step in 0..MAX_NEW_TOKENS {
            let embeddings = self.embed_text_tokens(&current_ids)?;

            // 디코더에 전달할 입력 형상: [batch, seq_len, hidden_dim]
            let embed_shape_usize: Vec<usize> = embeddings.shape.iter().map(|&s| s as usize).collect();
            let input_value = ort::value::Value::from_array(
                (embed_shape_usize, embeddings.data),
            )
            .map_err(|e| format!("Decoder input error: {}", e))?;

            let outputs = self
                .decoder
                .run(ort::inputs!["inputs_embeds" => input_value])
                .map_err(|e| format!("Decoder error: {}", e))?;

            // logits에서 마지막 토큰의 argmax 추출
            let (logits_shape, logits_data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Logits extract error: {}", e))?;

            // logits_shape: [batch, seq_len, vocab_size] (i64 슬라이스)
            let vocab_size = *logits_shape.last().unwrap_or(&1) as usize;
            let seq_len = if logits_shape.len() >= 2 {
                logits_shape[logits_shape.len() - 2] as usize
            } else {
                1
            };

            let last_logits_start = (seq_len - 1) * vocab_size;
            let last_logits = &logits_data[last_logits_start..last_logits_start + vocab_size];

            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx as i64)
                .ok_or("Empty logits")?;

            if next_token == eos_id {
                break;
            }

            generated_ids.push(next_token);
            current_ids = vec![next_token];
        }

        Ok(generated_ids)
    }
}
