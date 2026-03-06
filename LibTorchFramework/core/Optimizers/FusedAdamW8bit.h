#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/torch.h>
#include <torch/optim/optimizer.h>

struct FusedAdamW8bitOptions : public torch::optim::OptimizerCloneableOptions<FusedAdamW8bitOptions> 
{
    FusedAdamW8bitOptions(double lr = 1e-3);

    using betas_t = std::tuple<double, double>;

    TORCH_ARG(double, lr) = 1e-3;
    TORCH_ARG(betas_t, betas) = std::make_pair(0.9, 0.999);
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 1e-2;
    TORCH_ARG(bool, amsgrad) = false;

    TORCH_ARG(int64_t, block_size) = 256;
    TORCH_ARG(int64_t, min_quantized_numel) = 4096;
    TORCH_ARG(bool, bf16_stochastic_round) = false;

    // Fast-path tuning knobs.
    TORCH_ARG(int64_t, rescale_every) = 128;
    TORCH_ARG(double, overflow_factor) = 1.05;

    void serialize(torch::serialize::InputArchive& archive) override;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    TORCH_API friend bool operator==(const FusedAdamW8bitOptions& lhs, const FusedAdamW8bitOptions& rhs);
    double get_lr() const override;
    void set_lr(const double lr) override;
};

class FusedAdamW8bit : public torch::optim::Optimizer 
{
  public:
    explicit FusedAdamW8bit(const std::vector<torch::Tensor>& params, FusedAdamW8bitOptions defaults = {});
    explicit FusedAdamW8bit(
        std::vector<torch::optim::OptimizerParamGroup> param_groups,
        FusedAdamW8bitOptions defaults = {}
    );

    torch::Tensor step(torch::optim::Optimizer::LossClosure closure = nullptr) override;

    const FusedAdamW8bitOptions& options() const noexcept 
    {
        return *(static_cast<FusedAdamW8bitOptions*>(this->defaults_.get()));
    }

  private:
    struct QuantizedState 
    {
        torch::Tensor codes;  // uint8
        torch::Tensor absmax; // float32, per block
    };

    struct ParamState 
    {
        int64_t step = 0;
        bool quantized = false;
        QuantizedState exp_avg_q;
        QuantizedState exp_avg_sq_q;
        torch::Tensor exp_avg_fp32;
        torch::Tensor exp_avg_sq_fp32;
    };

    std::unordered_map<void*, ParamState> state_;
    torch::Tensor qmap_signed_cpu_;
    torch::Tensor qmap_unsigned_cpu_;

    static void validate_options(
        const FusedAdamW8bitOptions& options,
        const std::vector<torch::optim::OptimizerParamGroup>& param_groups
    );
    static torch::Tensor create_dynamic_map(bool signed_map, int max_exponent_bits = 7, int total_bits = 8);

    static std::pair<torch::Tensor, torch::Tensor> scale_tensor(const torch::Tensor& input, int64_t block_size);
    static torch::Tensor quantize_8bit_with_qmap(const torch::Tensor& input, const torch::Tensor& qmap);
    static torch::Tensor dequant_with_qmap(
        const torch::Tensor& codes,
        const torch::Tensor& qmap,
        const torch::Tensor& absmax,
        int64_t block_size
    );

    static QuantizedState quantize_from_fp32(
        const torch::Tensor& fp32,
        const torch::Tensor& qmap,
        int64_t block_size
    );
    static torch::Tensor dequant_to_fp32(const QuantizedState& qstate, const torch::Tensor& qmap, int64_t block_size);

    QuantizedState new_quantized_state(const torch::Tensor& param) const;
    ParamState& get_or_init_state(const torch::Tensor& param);
};
