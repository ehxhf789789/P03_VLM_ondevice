use ort::session::Session;
use ort::value::Value;
use std::path::Path;

use super::preprocessing;
use super::tokenizer::TokenizerWrapper;

/// 최대 생성 토큰 수
const MAX_NEW_TOKENS: usize = 512;

/// 디코더 레이어 수
const NUM_LAYERS: usize = 30;

/// KV cache 헤드 수
const NUM_KV_HEADS: usize = 3;

/// 헤드당 차원
const HEAD_DIM: usize = 64;

/// 비전 특징 벡터 (shape dims + flat data)
struct Features {
    shape: Vec<i64>,
    data: Vec<f32>,
}

/// KV cache 구조체
struct KvCache {
    /// past_key_values: [num_layers][2] where [0]=key, [1]=value
    /// 각각 shape: [batch=1, num_kv_heads=3, past_seq_len, head_dim=64]
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    past_seq_len: usize,
}

impl KvCache {
    /// 빈 KV cache를 생성한다.
    fn new() -> Self {
        Self {
            keys: vec![Vec::new(); NUM_LAYERS],
            values: vec![Vec::new(); NUM_LAYERS],
            past_seq_len: 0,
        }
    }

    /// 새로운 key/value를 기존 cache에 추가한다.
    fn extend(&mut self, layer: usize, new_key: &[f32], new_value: &[f32], new_seq_len: usize) {
        self.keys[layer].extend_from_slice(new_key);
        self.values[layer].extend_from_slice(new_value);
        if layer == NUM_LAYERS - 1 {
            self.past_seq_len += new_seq_len;
        }
    }
}

/// SmolVLM ONNX 3-세션 추론 엔진
pub struct SmolVlmEngine {
    vision_encoder: Session,
    embed_tokens: Session,
    decoder: Session,
    tokenizer: TokenizerWrapper,
}

impl SmolVlmEngine {
    /// ONNX 세션 빌더를 생성한다. Android에서는 스레드 수를 제한한다.
    fn create_session_builder() -> Result<ort::session::builder::SessionBuilder, ort::Error> {
        let builder = Session::builder()?;
        #[cfg(target_os = "android")]
        let builder = builder.with_intra_threads(2)?;
        Ok(builder)
    }

    /// 모델 디렉토리에서 세 개의 ONNX 세션을 로드한다.
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        log::info!("Loading ONNX sessions from {:?}", model_dir);

        let vision_encoder = Self::create_session_builder()
            .and_then(|b| b.commit_from_file(model_dir.join("vision_encoder.onnx")))
            .map_err(|e| format!("Failed to load vision_encoder: {}", e))?;

        let embed_tokens = Self::create_session_builder()
            .and_then(|b| b.commit_from_file(model_dir.join("embed_tokens.onnx")))
            .map_err(|e| format!("Failed to load embed_tokens: {}", e))?;

        let decoder = Self::create_session_builder()
            .and_then(|b| b.commit_from_file(model_dir.join("decoder_model_merged.onnx")))
            .map_err(|e| format!("Failed to load decoder: {}", e))?;

        let tokenizer = TokenizerWrapper::from_file(&model_dir.join("tokenizer.json"))?;

        // 모델 입력/출력 정보 로깅 (디버깅용)
        for input in vision_encoder.inputs() {
            log::info!("Vision encoder input: name={}", input.name());
        }
        for output in vision_encoder.outputs() {
            log::info!("Vision encoder output: name={}", output.name());
        }

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
        // pv_shape: [1, 3, H, W] → [1, 1, 3, H, W] (batch, num_images, channels, H, W)
        let h = pv_shape[2];
        let w = pv_shape[3];
        let pv_shape_5d = vec![1, 1, 3, h, w];
        let input_value = ort::value::Value::from_array(
            (pv_shape_5d, pv_data.to_vec()),
        )
        .map_err(|e| format!("Vision input error: {}", e))?;

        // pixel_attention_mask: [batch, num_images, H, W] = [1, 1, H, W] (bool type)
        let mask_data: Vec<bool> = vec![true; h * w];
        let mask_value = ort::value::Value::from_array(
            (vec![1usize, 1, h, w], mask_data),
        )
        .map_err(|e| format!("Vision mask input error: {}", e))?;

        let outputs = self
            .vision_encoder
            .run(ort::inputs![
                "pixel_values" => input_value,
                "pixel_attention_mask" => mask_value
            ])
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

    /// KV cache용 past_key_values 입력을 생성한다.
    fn build_kv_cache_inputs(
        &self,
        kv_cache: &KvCache,
    ) -> Result<Vec<(String, ort::value::DynValue)>, String> {
        let mut inputs = Vec::new();
        let past_seq_len = kv_cache.past_seq_len;

        for layer in 0..NUM_LAYERS {
            let key_name = format!("past_key_values.{}.key", layer);
            let value_name = format!("past_key_values.{}.value", layer);

            // Shape: [batch=1, num_kv_heads=3, past_seq_len, head_dim=64]
            let key_value: ort::value::DynValue = Value::from_array((
                vec![1usize, NUM_KV_HEADS, past_seq_len, HEAD_DIM],
                kv_cache.keys[layer].clone(),
            ))
            .map_err(|e| format!("KV cache key error (layer {}): {}", layer, e))?
            .into();

            let val_value: ort::value::DynValue = Value::from_array((
                vec![1usize, NUM_KV_HEADS, past_seq_len, HEAD_DIM],
                kv_cache.values[layer].clone(),
            ))
            .map_err(|e| format!("KV cache value error (layer {}): {}", layer, e))?
            .into();

            inputs.push((key_name, key_value));
            inputs.push((value_name, val_value));
        }

        Ok(inputs)
    }

    /// 디코더 출력에서 present_key_values를 추출하여 KV cache를 업데이트한다.
    fn update_kv_cache(
        outputs: &ort::session::SessionOutputs,
        kv_cache: &mut KvCache,
        new_seq_len: usize,
    ) -> Result<(), String> {
        // 출력 순서: logits, present.0.key, present.0.value, present.1.key, ...
        for layer in 0..NUM_LAYERS {
            let key_idx = 1 + layer * 2;
            let value_idx = 2 + layer * 2;

            let (_, key_data) = outputs[key_idx]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Extract present key error (layer {}): {}", layer, e))?;

            let (_, value_data) = outputs[value_idx]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Extract present value error (layer {}): {}", layer, e))?;

            // present는 전체 시퀀스를 포함하므로, 새로운 부분만 추출 대신 전체를 교체
            kv_cache.keys[layer] = key_data.to_vec();
            kv_cache.values[layer] = value_data.to_vec();
        }
        kv_cache.past_seq_len += new_seq_len;

        Ok(())
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
        let mut kv_cache = KvCache::new();

        for step in 0..MAX_NEW_TOKENS {
            let embeddings = self.embed_text_tokens(&current_ids)?;

            // 디코더에 전달할 입력 형상: [batch, seq_len, hidden_dim]
            let embed_shape_usize: Vec<usize> =
                embeddings.shape.iter().map(|&s| s as usize).collect();
            let seq_len = embed_shape_usize[1];
            let total_seq_len = kv_cache.past_seq_len + seq_len;

            let input_value = Value::from_array((embed_shape_usize, embeddings.data))
                .map_err(|e| format!("Decoder input error: {}", e))?;

            // attention_mask: [batch, total_seq_len] - 모든 토큰에 대해 1
            let attn_mask_data: Vec<i64> = vec![1i64; total_seq_len];
            let attn_mask_value = Value::from_array((vec![1usize, total_seq_len], attn_mask_data))
                .map_err(|e| format!("Decoder attention mask error: {}", e))?;

            // position_ids: [batch, seq_len] - 현재 위치부터 시작
            let current_pos = kv_cache.past_seq_len as i64;
            let position_ids_data: Vec<i64> =
                (current_pos..current_pos + seq_len as i64).collect();
            let position_ids_value =
                Value::from_array((vec![1usize, seq_len], position_ids_data))
                    .map_err(|e| format!("Decoder position_ids error: {}", e))?;

            // 기본 입력 구성 (DynValue로 변환)
            let mut input_map: Vec<(String, ort::value::DynValue)> = vec![
                ("inputs_embeds".to_string(), input_value.into()),
                ("attention_mask".to_string(), attn_mask_value.into()),
                ("position_ids".to_string(), position_ids_value.into()),
            ];

            // KV cache 입력 추가 (첫 번째 패스가 아닐 때만)
            if kv_cache.past_seq_len > 0 {
                let kv_inputs = self.build_kv_cache_inputs(&kv_cache)?;
                input_map.extend(kv_inputs);
            }

            // 입력 이름을 별도로 저장 (lifetime 문제 해결)
            let input_names: Vec<String> = input_map.iter().map(|(n, _)| n.clone()).collect();
            let input_values: Vec<ort::value::DynValue> =
                input_map.into_iter().map(|(_, v)| v).collect();

            // 입력을 ort::inputs! 매크로 형식으로 변환
            let inputs: Vec<(&str, ort::value::DynValue)> = input_names
                .iter()
                .map(|s| s.as_str())
                .zip(input_values)
                .collect();

            let outputs = self
                .decoder
                .run(inputs)
                .map_err(|e| format!("Decoder error (step {}): {}", step, e))?;

            // logits에서 마지막 토큰의 argmax 추출
            let (logits_shape, logits_data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Logits extract error: {}", e))?;

            let vocab_size = *logits_shape.last().unwrap_or(&1) as usize;
            let out_seq_len = if logits_shape.len() >= 2 {
                logits_shape[logits_shape.len() - 2] as usize
            } else {
                1
            };

            let last_logits_start = (out_seq_len - 1) * vocab_size;
            let last_logits = &logits_data[last_logits_start..last_logits_start + vocab_size];

            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx as i64)
                .ok_or("Empty logits")?;

            // KV cache 업데이트
            Self::update_kv_cache(&outputs, &mut kv_cache, seq_len)?;

            if next_token == eos_id {
                break;
            }

            generated_ids.push(next_token);
            current_ids = vec![next_token];
        }

        Ok(generated_ids)
    }
}
