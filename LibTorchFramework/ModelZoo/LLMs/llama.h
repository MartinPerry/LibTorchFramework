#ifndef LLAMA_MODEL_H
#define LLAMA_MODEL_H

#include <cstdint>
#include <utility>
#include <optional>
#include <vector>

#include <torch/torch.h>

#include "../../core/Modules/ModulesOptions.h"

#include "../../core/AbstractModel.h"

namespace ModelZoo
{
    namespace llama
    {
        struct KVCache
        {
            torch::Tensor k;
            torch::Tensor v;

            KVCache() = default;

            KVCache(torch::Tensor k, torch::Tensor v) : 
                k(std::move(k)), 
                v(std::move(v)) 
            {}

            bool defined() const noexcept
            {
                return k.defined() && v.defined();
            }

            void reset() noexcept
            {
                k = torch::Tensor();
                v = torch::Tensor();
            }
        };

        struct RMSNormImpl : torch::nn::Module 
        {
        public:            
            explicit RMSNormImpl(int64_t dim, double eps = 1e-6);
            torch::Tensor forward(const torch::Tensor& x);

        private:
            torch::Tensor weight;
            double eps;
        };
        TORCH_MODULE(RMSNorm);

        //========================================================================


        struct MLPImpl : torch::nn::Module 
        {            
        public:
            MLPImpl(int64_t dim, int64_t hidden_dim);
            torch::Tensor forward(const torch::Tensor& x);

        private:
            torch::nn::Linear gate_proj{ nullptr };
            torch::nn::Linear up_proj{ nullptr };
            torch::nn::Linear down_proj{ nullptr };

        };
        TORCH_MODULE(MLP);

        //========================================================================

        struct AttentionImpl : torch::nn::Module 
        {       
        public:
           
            AttentionImpl(int64_t dim, int64_t n_heads, std::optional<int64_t> n_kv_heads_opt = std::nullopt);
           
            std::pair<torch::Tensor, std::optional<KVCache>> forward(const torch::Tensor& x,
                const torch::Tensor& cos, const torch::Tensor& sin,
                const torch::Tensor& attn_mask, 
                const std::optional<KVCache>& past_kv = std::nullopt, 
                bool use_cache = false, 
                int64_t cache_position = 0);

        protected:            
            int64_t n_heads;
            int64_t n_kv_heads;
            int64_t head_dim;
            torch::nn::Linear q_proj{ nullptr };
            torch::nn::Linear k_proj{ nullptr };
            torch::nn::Linear v_proj{ nullptr };
            torch::nn::Linear o_proj{ nullptr };

            torch::Tensor apply_rope(const torch::Tensor& x, 
                const torch::Tensor& cos, const torch::Tensor& sin,
                int startPos = 0);
        };
        TORCH_MODULE(Attention);

        //========================================================================

        struct BlockImpl : torch::nn::Module 
        {
        public:
            
            BlockImpl(int64_t dim, int64_t n_heads, int64_t hidden_dim, std::optional<int64_t> n_kv_heads = std::nullopt,
                double rms_eps = 1e-6);

            std::pair<torch::Tensor, std::optional<KVCache>> forward(const torch::Tensor& x, 
                const torch::Tensor& cos, const torch::Tensor& sin,
                const torch::Tensor& attn_mask, 
                const std::optional<KVCache>& past_kv = std::nullopt,
                bool use_cache = false, 
                int64_t cache_position = 0);

        private:
            RMSNorm attn_norm{ nullptr };
            RMSNorm ffn_norm{ nullptr };
            Attention attn{ nullptr };
            MLP mlp{ nullptr };

        };
        TORCH_MODULE(Block);

        //========================================================================

        struct LlamaConfig 
        {
            int64_t vocab_size = 32000;
            int64_t hidden_size = 4096;
            int64_t num_hidden_layers = 32;
            int64_t num_attention_heads = 32;
            std::optional<int64_t> num_key_value_heads = std::nullopt;
            std::optional<int64_t> intermediate_size = std::nullopt;
            double rms_norm_eps = 1e-6;
            double rope_theta = 10000.0;
            bool tie_word_embeddings = true;

            static LlamaConfig FromJsonString(const std::string& jsonText);
            static LlamaConfig FromJsonFile(const std::string& filePath);

            static std::u8string InstructPrompt(std::u8string_view userText,
                std::u8string_view systemText = u8"You are a helpful assistant.");
        };

        //========================================================================

        struct LlamaForCausalLM : public AbstractModel
        {
        public:
                                   
            explicit LlamaForCausalLM(const LlamaConfig& cfg);

            const char* GetName() const override;

            const LlamaConfig& GetConfig() const;

            torch::Tensor get_attn_mask(int64_t q_len, int64_t k_len, torch::ScalarType dtype, int64_t past_len = 0);
            std::pair<torch::Tensor, torch::Tensor> get_rope(int64_t T, torch::ScalarType dtype);

            std::vector<torch::Tensor> RunForward(DataLoaderData& batch) override;

            torch::Tensor forward(const torch::Tensor& input_ids);

            std::pair<torch::Tensor, std::vector<KVCache>> forward_with_cache(const torch::Tensor& input_ids, 
                const std::vector<KVCache>& past_key_values,
                bool use_cache);

        protected:
            LlamaConfig cfg;            
                                                      
            torch::TensorOptions tOptDevice;

            torch::nn::Embedding tok_emb{ nullptr };
            torch::nn::ModuleList layers;
            RMSNorm norm{ nullptr };
            torch::nn::Linear lm_head{ nullptr };

            torch::Tensor _attn_mask_cache;
            torch::Tensor _rope_cos;
            torch::Tensor _rope_sin;
            int64_t _mask_len = 0;
            int64_t _rope_len = 0;

            std::pair<torch::Tensor, torch::Tensor> precompute_rope_frequencies(int64_t dim,
                int64_t max_seq_len,
                double base,
                torch::ScalarType);
            
        };
       
    }  // namespace llama
}

#endif
