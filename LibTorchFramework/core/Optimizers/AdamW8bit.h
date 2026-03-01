#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/torch.h>
#include <torch/optim/optimizer.h>

struct AdamW8bitOptions 
{
    double lr = 1e-3;
    std::pair<double, double> betas = {0.9, 0.999};
    double eps = 1e-8;
    double weight_decay = 1e-2;
    bool amsgrad = false;

    // Matches ao_optim defaults/behavior.
    int64_t block_size = 256;
    int64_t min_quantized_numel = 4096;
    bool bf16_stochastic_round = false;
};

class AdamW8bit : public torch::optim::Optimizer 
{
public:
    explicit AdamW8bit(std::vector<torch::Tensor> params, AdamW8bitOptions options = {});

    torch::Tensor step(torch::optim::Optimizer::LossClosure closure = nullptr) override;

    const AdamW8bitOptions& options() const noexcept 
    { 
        return options_; 
    }

private:
    struct QuantizedState 
    {
        torch::Tensor codes; // uint8
        torch::Tensor scale; // float32, one value per block
        torch::Tensor qmap;  // float32, 256 entries
    };

    struct ParamState 
    {
        int64_t step = 0;
        bool quantized = false;

        QuantizedState exp_avg_q;
        QuantizedState exp_avg_sq_q;
        QuantizedState max_exp_avg_sq_q;

        torch::Tensor exp_avg_fp32;
        torch::Tensor exp_avg_sq_fp32;
        torch::Tensor max_exp_avg_sq_fp32;
    };

    AdamW8bitOptions options_;
    std::unordered_map<void*, ParamState> state_;

    torch::Tensor qmap_signed_cpu_;
    torch::Tensor qmap_unsigned_cpu_;

    static void validate_options(const AdamW8bitOptions& options);
    static torch::Tensor create_dynamic_map(bool signed_map, int max_exponent_bits = 7, int total_bits = 8);

    static std::pair<torch::Tensor, torch::Tensor> scale_tensor(const torch::Tensor& input, int64_t block_size);
    static torch::Tensor quantize_8bit_with_qmap(const torch::Tensor& input, const torch::Tensor& qmap);
    static torch::Tensor dequant_with_qmap(const torch::Tensor& codes, const torch::Tensor& qmap, const torch::Tensor& scale);

    static QuantizedState quantize_from_fp32(const torch::Tensor& fp32, const torch::Tensor& qmap, int64_t block_size);
    static torch::Tensor dequant_to_fp32(const QuantizedState& qstate);

    QuantizedState new_quantized_state(const torch::Tensor& param, bool signed_map) const;
    ParamState& get_or_init_state(const torch::Tensor& param);
    void save_state_tensor(ParamState& state, bool is_first_moment, const torch::Tensor& value_fp32);
    void save_max_state_tensor(ParamState& state, const torch::Tensor& value_fp32);   
};

