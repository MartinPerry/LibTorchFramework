#ifndef SDVAE_ATTENTION_H
#define SDVAE_ATTENTION_H

#include <torch/torch.h>
#include <cmath>
#include <limits>

// SelfAttention and CrossAttention implemented with LibTorch (C++ API).
// Note: some helper/stub functionality (e.g., detailed validation, optional dtype handling)
// can be added later as needed.

namespace ModelZoo {
    namespace sdvae {

        class SelfAttentionImpl : public torch::nn::Module
        {
        public:
            SelfAttentionImpl(int64_t n_heads, int64_t d_embed, bool in_proj_bias = true, bool out_proj_bias = true);

            // x: (batch, seq_len, dim)
            // causal_mask: if true, apply causal masking (upper triangle masked)
            torch::Tensor forward(torch::Tensor x, bool causal_mask = false);

        private:
            torch::nn::Linear in_proj{ nullptr };   // projects to 3 * d_embed
            torch::nn::Linear out_proj{ nullptr };  // projects back to d_embed
            int64_t n_heads;
            int64_t d_head;
        };

        TORCH_MODULE(SelfAttention);


        class CrossAttentionImpl
            : public torch::nn::Module
        {
        public:
            CrossAttentionImpl(int64_t n_heads, int64_t d_embed, int64_t d_cross, bool in_proj_bias = true, bool out_proj_bias = true);

            // x: (batch, seq_len_q, dim_q)
            // y: (batch, seq_len_kv, dim_kv)
            torch::Tensor forward(torch::Tensor x, torch::Tensor y);

        private:
            torch::nn::Linear q_proj{ nullptr };
            torch::nn::Linear k_proj{ nullptr };
            torch::nn::Linear v_proj{ nullptr };
            torch::nn::Linear out_proj{ nullptr };
            int64_t n_heads;
            int64_t d_head;
        };

        TORCH_MODULE(CrossAttention);
    }
}

#endif