use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        println!("Before increment: {}", cache.len());
        cache.increment(seq_len);
        println!("After increment: {}", cache.len());

        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;
        println!(
            "n_q_h: {}, n_kv_h: {}, n_groups: {}, dqkv: {}, hidden_size: {}",
            self.n_q_h, self.n_kv_h, n_groups, self.dqkv, self.d
        );

        assert_eq!(
            self.d,
            self.n_q_h * self.dqkv,
            "hidden_size mismatch with n_q_h * dqkv"
        );
        assert_eq!(
            self.d,
            self.n_kv_h * n_groups * self.dqkv,
            "hidden_size mismatch with n_kv_h * n_groups * dqkv"
        );

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        println!("q_buf.shape: {:?}", q_buf.shape());
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]);

            let k = &mut cache.k_cache(layer, past_seq_len);
            let v = &mut cache.v_cache(layer, past_seq_len);
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            println!("q.shape after matmul_transb: {:?}", q.shape());
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            println!(
                "n_q_h: {}, n_kv_h: {}, n_groups: {}, dqkv: {}, hidden_size: {}",
                self.n_q_h, self.n_kv_h, n_groups, self.dqkv, self.d
            );
            let q = q.reshape(&vec![seq_len, self.n_kv_h * n_groups * self.dqkv]);

            let full_k = &mut cache.k_cache(layer, 0);
            let full_v = &mut cache.v_cache(layer, 0);
            // println!("q shape length: {:?}", q.shape());
            // println!("q_data length: {}", q.data().len());
            // println!("k_data length: {}", k.data().len());
            // println!("att_scores_data length: {}", att_scores.size());

            // Call self_attention

            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );

            OP::matmul_transb(
                &mut residual,
                1.0,
                &hidden_states,
                &self.params.wo[layer],
                1.0,
            );
            println!("FUCK!hidden_states shape: {:?}", hidden_states.shape());
            println!("residual shape: {:?}", residual.shape());
            println!("rms_out_w shape: {:?}", self.params.rms_out_w.shape());
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        let mut logits = Tensor::<f32>::default(&vec![seq_len, self.vocab]);

        // 確保 hidden_states 的形狀與 residual 一致
        let mut hidden_states_slice = hidden_states.slice(0, &vec![seq_len, self.d]);
        let residual_slice = residual.slice(0, &vec![seq_len, self.d]);

        println!(
            "hidden_states_slice shape: {:?}",
            hidden_states_slice.shape()
        );
        println!("residual shape: {:?}", residual_slice.shape());
        println!("rms_out_w shape: {:?}", self.params.rms_out_w.shape());

        OP::rms_norm(
            &mut hidden_states_slice,
            &residual_slice,
            &self.params.rms_out_w,
            self.eps,
        );

        println!(
            "hidden_states_slice shape after rms_norm: {:?}",
            hidden_states_slice.shape()
        );
        println!("logits shape after rms_norm: {:?}", logits.shape());

        OP::matmul_transb(
            &mut logits,
            0.,
            &hidden_states_slice,
            &self.params.lm_head,
            1.0,
        );

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        // 初始化輸出序列，包含 BOS token
        let mut result = token_ids.to_vec();
        if result.is_empty() {
            result.push(self.bos_token_id); // 確保有初始 token
        }

        // 初始化 KVCache 和輸入張量
        let mut kvcache = self.new_cache();
        let mut input_tensors = Tensor::<u32>::new(result.clone(), &vec![result.len()]);
        let mut cnt = 0;
        // 開始生成過程
        while result.len() < max_len {
            println!("Current result: {:?}", result);

            // 前向傳播
            let logits = self.forward(&input_tensors, &mut kvcache);
            cnt += 1;
            print!("COUNT: {:?}\n", cnt);
            // 取樣下一個 token
            println!("logits shape before random_sample: {:?}", logits.shape());
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            println!("Generated token: {}", next_token);

            result.push(next_token);

            // 終止條件：生成 EOS token
            if next_token == self.eos_token_id {
                break;
            }

            // 更新輸入張量，僅包含當前生成的 token
            input_tensors = Tensor::<u32>::new(vec![next_token], &vec![1]);
        }

        result
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    println!("n_kv_h: {}, n_groups: {}, dqkv: {}", n_kv_h, n_groups, dqkv);
    assert_eq!(
        q.shape(),
        &[seq_len, n_kv_h * n_groups * dqkv],
        "q shape is incorrect"
    );
    assert_eq!(
        k.shape(),
        &[total_seq_len, n_kv_h * dqkv],
        "k shape is incorrect"
    );
    assert_eq!(
        v.shape(),
        &[total_seq_len, n_kv_h * dqkv],
        "v shape is incorrect"
    );
    assert_eq!(
        hidden_states.shape(),
        &[seq_len, n_kv_h * n_groups * dqkv],
        "hidden_states shape is incorrect"
    );
    assert_eq!(
        att_scores.shape(),
        &[n_kv_h, n_groups, seq_len, total_seq_len],
        "att_scores shape is incorrect"
    );

    // 檢查數據長度是否合理
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();
    let att_scores_data = unsafe { att_scores.data_mut() };

    assert_eq!(
        q_data.len(),
        seq_len * n_kv_h * n_groups * dqkv,
        "q_data length is incorrect"
    );
    assert_eq!(
        k_data.len(),
        total_seq_len * n_kv_h * dqkv,
        "k_data length is incorrect"
    );
    assert_eq!(
        v_data.len(),
        total_seq_len * n_kv_h * dqkv,
        "v_data length is incorrect"
    );
    assert_eq!(
        att_scores_data.len(),
        n_kv_h * n_groups * seq_len * total_seq_len,
        "att_scores_data length is incorrect"
    );
    // Step 1: Compute scaled dot-product attention scores in a separate scope to avoid borrow conflicts
    {
        let num_heads = n_kv_h * n_groups;
        let q_data = q.data();
        let k_data = k.data();
        let mut att_scores_data = unsafe { att_scores.data_mut() };
        let scaling_factor = 1.0 / (dqkv as f32).sqrt();

        for kv_head in 0..n_kv_h {
            for group in 0..n_groups {
                let head_offset = (kv_head * n_groups + group) * dqkv;
                for seq_idx in 0..seq_len {
                    let q_offset = seq_idx * num_heads * dqkv + head_offset;
                    if q_offset + dqkv <= q_data.len() {
                        let q_vec = &q_data[q_offset..q_offset + dqkv];
                        for total_seq_idx in 0..total_seq_len {
                            let k_offset = total_seq_idx * n_kv_h * dqkv + kv_head * dqkv;
                            if k_offset + dqkv <= k_data.len() {
                                let k_vec = &k_data[k_offset..k_offset + dqkv];
                                let score: f32 = q_vec.iter().zip(k_vec).map(|(q, k)| q * k).sum();
                                let att_score_idx = kv_head * n_groups * seq_len * total_seq_len
                                    + group * seq_len * total_seq_len
                                    + seq_idx * total_seq_len
                                    + total_seq_idx;
                                if att_score_idx <= att_scores_data.len() {
                                    print!(
                                        "att_scores_data[{:?}]: {:?}\n",
                                        att_score_idx, att_scores_data[att_score_idx]
                                    );
                                    att_scores_data[att_score_idx] = score * scaling_factor;
                                } else {
                                    panic!(
                                        "Index out of bounds: att_score_idx = {}, len = {}",
                                        att_score_idx,
                                        att_scores_data.len()
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 2: Apply softmax to attention scores
    OP::masked_softmax(att_scores);

    // Step 3: Compute weighted sum of values (output) in another separate scope
    {
        let num_heads = n_kv_h * n_groups;
        let v_data = v.data();
        let att_scores_data = att_scores.data();
        let mut hidden_states_data = unsafe { hidden_states.data_mut() };

        for kv_head in 0..n_kv_h {
            for group in 0..n_groups {
                let head_offset = (kv_head * n_groups + group) * dqkv;
                for seq_idx in 0..seq_len {
                    let mut output_vec = vec![0.0; dqkv];
                    for total_seq_idx in 0..total_seq_len {
                        let att_score_idx = kv_head * n_groups * seq_len * total_seq_len
                            + group * seq_len * total_seq_len
                            + seq_idx * total_seq_len
                            + total_seq_idx;
                        if att_score_idx < att_scores_data.len() {
                            let att_score = att_scores_data[att_score_idx];

                            let v_offset = total_seq_idx * n_kv_h * dqkv + kv_head * dqkv;
                            if v_offset + dqkv <= v_data.len() {
                                let v_vec = &v_data[v_offset..v_offset + dqkv];
                                for i in 0..dqkv {
                                    output_vec[i] += att_score * v_vec[i];
                                }
                            }
                        }
                    }

                    // Write the output back to hidden states
                    let hidden_offset = seq_idx * num_heads * dqkv + head_offset;
                    if hidden_offset + dqkv <= hidden_states_data.len() {
                        hidden_states_data[hidden_offset..hidden_offset + dqkv]
                            .copy_from_slice(&output_vec);
                    }
                }
            }
        }
    }
}

// fn self_attention(
//     hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
//     att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
//     q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
//     k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
//     v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
//     n_kv_h: usize,
//     n_groups: usize,
//     seq_len: usize,
//     total_seq_len: usize,
//     dqkv: usize,
// ) {
//
// }

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    // todo!("Implement mlp");
    OP::rms_norm(hidden_states, &residual, &rms_w, eps);
    OP::matmul_transb(gate, 0., &hidden_states, &w_gate, 1.0);
    OP::matmul_transb(up, 0., &hidden_states, &w_up, 1.0);
    OP::swiglu(up, &gate);
    OP::matmul_transb(residual, 1.0, &up, &w_down, 1.0);
}
#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use crate::tensor::float_eq;
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(
        &model.params.embedding_table.data()[50],
        &0.14453125,
        1e-6
    ));
    assert_eq!(
        model.params.lm_head.data()[10],
        model.params.embedding_table.data()[10]
    );
    assert!(float_eq(
        &model.params.rms_att_w[0].data()[10],
        &0.18652344,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_ffn_w[1].data()[10],
        &0.32421875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_out_w.data()[100],
        &0.73046875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.w_down[0].data()[100],
        &-0.0625,
        1e-6
    ));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(
        &model.params.w_gate[1].data()[100],
        &0.296875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wq[1].data()[100],
        &0.032226563,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wk[1].data()[100],
        &-0.21386719,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wv[0].data()[100],
        &0.041015625,
        1e-6
    ));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
}
