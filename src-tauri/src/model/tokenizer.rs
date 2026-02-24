use std::path::Path;
use tokenizers::Tokenizer;

pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
}

impl TokenizerWrapper {
    /// tokenizer.json 파일에서 토크나이저를 로드한다.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        Ok(Self { tokenizer })
    }

    /// 텍스트를 토큰 ID 시퀀스로 인코딩한다.
    pub fn encode(&self, text: &str) -> Result<Vec<i64>, String> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| format!("Encoding error: {}", e))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
    }

    /// 토큰 ID 시퀀스를 텍스트로 디코딩한다.
    pub fn decode(&self, ids: &[i64]) -> Result<String, String> {
        let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.tokenizer
            .decode(&u32_ids, true)
            .map_err(|e| format!("Decoding error: {}", e))
    }

    /// EOS 토큰 ID를 반환한다.
    pub fn eos_token_id(&self) -> i64 {
        // SmolVLM의 EOS 토큰 ID (일반적으로 2)
        self.tokenizer
            .token_to_id("</s>")
            .or_else(|| self.tokenizer.token_to_id("<|im_end|>"))
            .unwrap_or(2) as i64
    }

    /// 어휘 크기를 반환한다.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}
