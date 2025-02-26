use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        // let get_tensor: impl Fn(&str) -> Tensor<f32> = |name: &str| {
        // ...    
        // };
        
        // LLamaParams {
        //     embedding_table: get_tensor(...),
        //     ...
        // }
        // {
        //     "bos_token_id": 1, # 起始符token id
        //     "eos_token_id": 2, # 结束符token id
        //     "hidden_size": 128, # 隐藏层大小，即各层输出的最后一维
        //     "intermediate_size": 384, # Feed-Forward神经网络的中间层大小
        //     "max_position_embeddings": 512, # 最大序列长度
        //     "num_attention_heads": 8, # Self-Attention的Q头数
        //     "num_hidden_layers": 2, # 隐藏层数
        //     "num_key_value_heads": 4, # Self-Attention的K和V头数
        //     "rms_norm_eps": 1e-6, # RMS Normalization的epsilon参数
        //     "rope_theta": 10000.0, # RoPE的theta参数
        //     "tie_word_embeddings": true, # 起始和结束embedding参数矩阵是否共享同一份数据
        //     "torch_dtype": "float32", # 模型数据类型
        //     "vocab_size": 2048 # 词表大小
        //   }
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap();
            let data = unsafe {
                let ptr = tensor.data().as_ptr() as *const f32;
                let len = tensor.data().len() / std::mem::size_of::<f32>();
                std::slice::from_raw_parts(ptr, len).to_vec()
            };
            
            Tensor::new(data, &tensor.shape().to_vec())
        };

        let n_layers = config.num_hidden_layers;
        let load_weight = |param_suffix: &str| -> Vec<Tensor<f32>> {
            (0..n_layers)
                .map(|layer_idx| get_tensor(&format!("model.layers.{}.{}", layer_idx, param_suffix)))
                .collect()
        };
        LLamaParams {
            // 输入层
            embedding_table: get_tensor("lm_head.weight"),  // 注意：根据网页7[7](@ref)可能需要改为 model.embed_tokens.weight
            
            // 注意力层
            rms_att_w: load_weight("input_layernorm.weight"),
            wq: load_weight("self_attn.q_proj.weight"),
            wk: load_weight("self_attn.k_proj.weight"),
            wv: load_weight("self_attn.v_proj.weight"),
            wo: load_weight("self_attn.o_proj.weight"),
            
            // FFN层
            rms_ffn_w: load_weight("post_attention_layernorm.weight"),
            w_up: load_weight("mlp.up_proj.weight"),
            w_gate: load_weight("mlp.gate_proj.weight"),
            w_down: load_weight("mlp.down_proj.weight"),
            
            // 输出层
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),  // 网页7[7](@ref)建议检查参数共享逻辑
        }
    }
}
