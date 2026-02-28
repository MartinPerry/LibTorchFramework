#include "AdamW8bit.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#include <torch/optim/adamw.h>



using torch::indexing::Slice;

AdamW8bit::AdamW8bit(std::vector<torch::Tensor> params, AdamW8bitOptions options)
    : torch::optim::Optimizer(
          std::move(params),
          std::make_unique<torch::optim::AdamWOptions>(options.lr)
      ),
      options_(std::move(options)) {
    validate_options(options_);

    if (param_groups().empty()) {
        throw std::invalid_argument("AdamW8bit requires at least one parameter tensor.");
    }

    bool any_param = false;
    for (const auto& group : param_groups()) {
        for (const auto& p : group.params()) {
            any_param = true;
            if (!p.defined()) {
                throw std::invalid_argument("AdamW8bit received an undefined parameter tensor.");
            }
            if (!p.is_leaf()) {
                throw std::invalid_argument("AdamW8bit expects leaf tensors as parameters.");
            }
            if (!p.is_floating_point()) {
                throw std::invalid_argument("AdamW8bit expects floating-point parameter tensors.");
            }
        }
    }
    if (!any_param) {
        throw std::invalid_argument("AdamW8bit requires at least one parameter tensor.");
    }

    qmap_signed_cpu_ = create_dynamic_map(true);
    qmap_unsigned_cpu_ = create_dynamic_map(false);
}

torch::Tensor AdamW8bit::step(torch::optim::Optimizer::LossClosure closure) {
    torch::Tensor loss;
    if (closure) {
        torch::AutoGradMode enable_grad(true);
        loss = closure();
    }
    torch::NoGradGuard no_grad;

    const double beta1 = options_.betas.first;
    const double beta2 = options_.betas.second;
    const double lr = options_.lr;
    const double eps = options_.eps;
    const double weight_decay = options_.weight_decay;

    for (auto& group : param_groups()) {
        for (auto& p : group.params()) {
            auto grad = p.grad();
            if (!grad.defined()) {
                continue;
            }
            if (grad.is_sparse()) {
                throw std::invalid_argument("AdamW8bit does not support sparse gradients.");
            }
            if (!grad.is_floating_point()) {
                throw std::invalid_argument("AdamW8bit expects floating-point gradients.");
            }
            if (p.device() != grad.device()) {
                throw std::invalid_argument("Parameter and gradient must be on the same device.");
            }

            auto& state = get_or_init_state(p);
            state.step += 1;
            const double step_d = static_cast<double>(state.step);

            auto grad_f32 = grad.contiguous().to(torch::kFloat32);
            auto p_f32 = p.contiguous().to(torch::kFloat32);

            // AdamW decoupled weight decay (ao_optim's IS_ADAMW=True path).
            if (weight_decay != 0.0) {
                p_f32 = p_f32 - (lr * weight_decay) * p_f32;
            }

            auto exp_avg_f32 = state.quantized ? dequant_to_fp32(state.exp_avg_q) : state.exp_avg_fp32;
            auto exp_avg_sq_f32 = state.quantized ? dequant_to_fp32(state.exp_avg_sq_q) : state.exp_avg_sq_fp32;

            exp_avg_f32 = exp_avg_f32.lerp(grad_f32, 1.0 - beta1);
            exp_avg_sq_f32 = exp_avg_sq_f32.lerp(grad_f32.square(), 1.0 - beta2);

            save_state_tensor(state, true, exp_avg_f32);
            save_state_tensor(state, false, exp_avg_sq_f32);

            torch::Tensor denom_base;
            if (options_.amsgrad) {
                auto max_exp_avg_sq_f32 =
                    state.quantized ? dequant_to_fp32(state.max_exp_avg_sq_q) : state.max_exp_avg_sq_fp32;
                max_exp_avg_sq_f32 = torch::maximum(max_exp_avg_sq_f32, exp_avg_sq_f32);
                save_max_state_tensor(state, max_exp_avg_sq_f32);
                denom_base = max_exp_avg_sq_f32;
            } else {
                denom_base = exp_avg_sq_f32;
            }

            const double bias_correction1 = 1.0 - std::pow(beta1, step_d);
            const double bias_correction2 = 1.0 - std::pow(beta2, step_d);
            auto denom = (denom_base.sqrt() / std::sqrt(bias_correction2)).add(eps);
            auto new_p_f32 = p_f32 - lr * (exp_avg_f32 / bias_correction1) / denom;

            if (options_.bf16_stochastic_round && p.scalar_type() == torch::kBFloat16) {
                // Keep API parity with ao_optim; this C++ path currently falls back to deterministic cast.
                p.copy_(new_p_f32.to(torch::kBFloat16));
            } else {
                p.copy_(new_p_f32.to(p.scalar_type()));
            }
        }
    }

    return loss;
}

void AdamW8bit::validate_options(const AdamW8bitOptions& options) {
    if (options.lr < 0.0) {
        throw std::invalid_argument("Invalid learning rate.");
    }
    if (options.eps < 0.0) {
        throw std::invalid_argument("Invalid epsilon value.");
    }
    if (options.weight_decay < 0.0) {
        throw std::invalid_argument("Invalid weight_decay value.");
    }
    if (options.betas.first < 0.0 || options.betas.first >= 1.0 || options.betas.second < 0.0 ||
        options.betas.second >= 1.0) {
        throw std::invalid_argument("Invalid beta values.");
    }
    if (options.block_size < 1) {
        throw std::invalid_argument("block_size must be >= 1.");
    }
    if (options.min_quantized_numel < 1) {
        throw std::invalid_argument("min_quantized_numel must be >= 1.");
    }
}

torch::Tensor AdamW8bit::create_dynamic_map(bool signed_map, int max_exponent_bits, int total_bits) {
    std::vector<float> data;
    data.reserve(static_cast<size_t>(1 << total_bits));

    // Mirrors ao_optim/quant_utils.py:create_dynamic_map.
    const int non_sign_bits = total_bits - 1;
    const int additional_items = (1 << (non_sign_bits - max_exponent_bits)) - 1;
    int last_i = 0;

    for (int i = 0; i < max_exponent_bits; ++i) {
        last_i = i;
        const int fraction_items =
            signed_map ? ((1 << (i + non_sign_bits - max_exponent_bits)) + 1)
                       : ((1 << (i + non_sign_bits - max_exponent_bits + 1)) + 1);

        auto boundaries = torch::linspace(0.1, 1.0, fraction_items, torch::TensorOptions().dtype(torch::kFloat32));
        auto means = (boundaries.index({Slice(torch::indexing::None, -1)}) + boundaries.index({Slice(1, torch::indexing::None)})) / 2.0;
        const double scale = std::pow(10.0, -(max_exponent_bits - 1) + i);
        auto vals = (means * scale).contiguous().cpu();

        const auto* ptr = vals.data_ptr<float>();
        for (int64_t k = 0; k < vals.numel(); ++k) {
            data.push_back(ptr[k]);
        }
        if (signed_map) {
            for (int64_t k = 0; k < vals.numel(); ++k) {
                data.push_back(-ptr[k]);
            }
        }
    }

    if (additional_items > 0) {
        auto boundaries = torch::linspace(0.1, 1.0, additional_items + 1, torch::TensorOptions().dtype(torch::kFloat32));
        auto means = (boundaries.index({Slice(torch::indexing::None, -1)}) + boundaries.index({Slice(1, torch::indexing::None)})) / 2.0;
        const double scale = std::pow(10.0, -(max_exponent_bits - 1) + last_i);
        auto vals = (means * scale).contiguous().cpu();

        const auto* ptr = vals.data_ptr<float>();
        for (int64_t k = 0; k < vals.numel(); ++k) {
            data.push_back(ptr[k]);
        }
        if (signed_map) {
            for (int64_t k = 0; k < vals.numel(); ++k) {
                data.push_back(-ptr[k]);
            }
        }
    }

    data.push_back(0.0f);
    data.push_back(1.0f);
    std::sort(data.begin(), data.end());

    if (static_cast<int64_t>(data.size()) != (1LL << total_bits)) {
        throw std::runtime_error("create_dynamic_map produced unexpected table size.");
    }

    return torch::tensor(data, torch::TensorOptions().dtype(torch::kFloat32));
}

std::pair<torch::Tensor, torch::Tensor> AdamW8bit::scale_tensor(const torch::Tensor& input, int64_t block_size) {
    const auto numel = input.numel();
    if (numel % block_size != 0) {
        throw std::invalid_argument("scale_tensor expects input.numel() divisible by block_size.");
    }
    auto in = input.contiguous().to(torch::kFloat32);
    auto reshaped = in.view({-1, block_size});
    auto scale = std::get<0>(reshaped.abs().max(-1, false)).clamp_min(1e-12);
    auto scaled = reshaped / scale.view({-1, 1});
    return {scaled.view(input.sizes()).contiguous(), scale.contiguous()};
}

torch::Tensor AdamW8bit::quantize_8bit_with_qmap(const torch::Tensor& input, const torch::Tensor& qmap) {
    auto in = input.contiguous().to(torch::kFloat32);
    auto map = qmap.to(in.device(), torch::kFloat32);

    auto codes = torch::where(
        in >= map.index({128}),
        torch::full(in.sizes(), 128, in.options().dtype(torch::kInt32)),
        torch::zeros(in.sizes(), in.options().dtype(torch::kInt32))
    );

    const int bits[] = {64, 32, 16, 8, 4, 2, 1};
    for (int bit : bits) {
        auto idx = (codes + bit).to(torch::kLong);
        auto thresh = map.index({idx});
        auto add = torch::where(in >= thresh, torch::full_like(codes, bit), torch::zeros_like(codes));
        codes = codes + add;
    }

    auto codes_up = (codes + 1).clamp_max(255);
    auto val_down = map.index({codes.to(torch::kLong)});
    auto val_up = map.index({codes_up.to(torch::kLong)});
    auto residual = in - val_down;
    codes = torch::where(residual >= (val_up - val_down) * 0.5, codes_up, codes);

    return codes.to(torch::kUInt8).contiguous();
}

torch::Tensor AdamW8bit::dequant_with_qmap(const torch::Tensor& codes, const torch::Tensor& qmap, const torch::Tensor& scale) {
    auto map = qmap.to(codes.device(), torch::kFloat32);
    auto out = map.index({codes.to(torch::kLong)}).view({scale.size(0), -1}) * scale.view({-1, 1});
    return out.view(codes.sizes()).contiguous();
}

AdamW8bit::QuantizedState AdamW8bit::quantize_from_fp32(
    const torch::Tensor& fp32,
    const torch::Tensor& qmap,
    int64_t block_size
) {
    auto [scaled, scale] = scale_tensor(fp32, block_size);
    auto codes = quantize_8bit_with_qmap(scaled, qmap);
    return {codes, scale, qmap};
}

torch::Tensor AdamW8bit::dequant_to_fp32(const QuantizedState& qstate) {
    return dequant_with_qmap(qstate.codes, qstate.qmap, qstate.scale).to(torch::kFloat32);
}

AdamW8bit::QuantizedState AdamW8bit::new_quantized_state(const torch::Tensor& param, bool signed_map) const {
    const int64_t blocks = param.numel() / options_.block_size;
    auto qmap = (signed_map ? qmap_signed_cpu_ : qmap_unsigned_cpu_).to(param.device(), torch::kFloat32);
    auto codes = torch::zeros(param.sizes(), param.options().dtype(torch::kUInt8));
    auto scale = torch::zeros({blocks}, param.options().dtype(torch::kFloat32));
    return {codes.contiguous(), scale.contiguous(), qmap.contiguous()};
}

AdamW8bit::ParamState& AdamW8bit::get_or_init_state(const torch::Tensor& param) {
    auto* key = static_cast<void*>(param.unsafeGetTensorImpl());
    auto it = state_.find(key);
    if (it != state_.end()) {
        return it->second;
    }

    ParamState s;
    s.quantized = (param.numel() >= options_.min_quantized_numel) && (param.numel() % options_.block_size == 0);

    if (s.quantized) {
        s.exp_avg_q = new_quantized_state(param, true);
        s.exp_avg_sq_q = new_quantized_state(param, false);
        if (options_.amsgrad) {
            s.max_exp_avg_sq_q = new_quantized_state(param, false);
        }
    } else {
        auto fp32_opts = param.options().dtype(torch::kFloat32);
        s.exp_avg_fp32 = torch::zeros(param.sizes(), fp32_opts);
        s.exp_avg_sq_fp32 = torch::zeros(param.sizes(), fp32_opts);
        if (options_.amsgrad) {
            s.max_exp_avg_sq_fp32 = torch::zeros(param.sizes(), fp32_opts);
        }
    }

    auto [inserted_it, inserted] = state_.emplace(key, std::move(s));
    (void)inserted;
    return inserted_it->second;
}

void AdamW8bit::save_state_tensor(ParamState& state, bool is_first_moment, const torch::Tensor& value_fp32) {
    if (state.quantized) {
        if (is_first_moment) {
            auto q = quantize_from_fp32(value_fp32, state.exp_avg_q.qmap, options_.block_size);
            state.exp_avg_q.codes = q.codes;
            state.exp_avg_q.scale = q.scale;
        } else {
            auto q = quantize_from_fp32(value_fp32, state.exp_avg_sq_q.qmap, options_.block_size);
            state.exp_avg_sq_q.codes = q.codes;
            state.exp_avg_sq_q.scale = q.scale;
        }
    } else {
        if (is_first_moment) {
            state.exp_avg_fp32 = value_fp32;
        } else {
            state.exp_avg_sq_fp32 = value_fp32;
        }
    }
}

void AdamW8bit::save_max_state_tensor(ParamState& state, const torch::Tensor& value_fp32) {
    if (state.quantized) {
        auto q = quantize_from_fp32(value_fp32, state.max_exp_avg_sq_q.qmap, options_.block_size);
        state.max_exp_avg_sq_q.codes = q.codes;
        state.max_exp_avg_sq_q.scale = q.scale;
    } else {
        state.max_exp_avg_sq_fp32 = value_fp32;
    }
}


