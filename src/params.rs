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
        // Implement the logic to load model parameters from safetensors file
        let get_tensor = |name: &str| -> Tensor<f32> {
            // Implement the logic to retrieve the tensor by name
            let tensor_data: safetensors::tensor::TensorView<'_> = safetensor.tensor(name).expect(&format!("Tensor {} not found", name));
            Tensor::from(tensor_data.to_owned())
        };

        let get_layer_tensor = |layer: usize, name: &str| -> Tensor<f32> {
            let full_name = format!("model.layers.{}.{}", layer, name);
            get_tensor(&full_name)
        };

        let num_layers = config.num_hidden_layers; // Use the correct field name
        let mut rms_att_w = Vec::with_capacity(num_layers);
        let mut wq = Vec::with_capacity(num_layers);
        let mut wk = Vec::with_capacity(num_layers);
        let mut wv = Vec::with_capacity(num_layers);
        let mut wo = Vec::with_capacity(num_layers);
        let mut rms_ffn_w = Vec::with_capacity(num_layers);
        let mut w_up = Vec::with_capacity(num_layers);
        let mut w_gate = Vec::with_capacity(num_layers);
        let mut w_down = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            rms_att_w.push(get_layer_tensor(layer, "input_layernorm.weight"));
            wq.push(get_layer_tensor(layer, "attention.query.weight"));
            wk.push(get_layer_tensor(layer, "attention.key.weight"));
            wv.push(get_layer_tensor(layer, "attention.value.weight"));
            wo.push(get_layer_tensor(layer, "attention.output.weight"));
            rms_ffn_w.push(get_layer_tensor(layer, "ffn_layernorm.weight"));
            w_up.push(get_layer_tensor(layer, "ffn.up_proj.weight"));
            w_gate.push(get_layer_tensor(layer, "ffn.gate_proj.weight"));
            w_down.push(get_layer_tensor(layer, "ffn.down_proj.weight"));
        }
        
        LLamaParams {
            embedding_table: get_tensor("embedding_table"),
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w: get_tensor("rms_out_w"),
            lm_head: get_tensor("lm_head"),
        }
    }
}
