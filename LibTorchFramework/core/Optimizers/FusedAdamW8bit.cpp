#include "./FusedAdamW8bit.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "./FusedAdamW8bit_cuda.h"

using torch::indexing::Slice;

#if !defined(USE_CUDA)
void adamw8bit_fused_cuda_step(
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    int64_t,
    double,
    double,
    double,
    double,
    double,
    double,
    int64_t,
    int64_t,
    double
) 
{
    throw std::runtime_error("FusedAdamW8bit was built without CUDA support.");
}
#endif

FusedAdamW8bitOptions::FusedAdamW8bitOptions(double lr) : 
    lr_(lr) 
{
}

bool operator==(const FusedAdamW8bitOptions& lhs, const FusedAdamW8bitOptions& rhs) 
{
    return (lhs.lr() == rhs.lr()) && (std::get<0>(lhs.betas()) == std::get<0>(rhs.betas())) &&
           (std::get<1>(lhs.betas()) == std::get<1>(rhs.betas())) && (lhs.eps() == rhs.eps()) &&
           (lhs.weight_decay() == rhs.weight_decay()) && (lhs.amsgrad() == rhs.amsgrad()) &&
           (lhs.block_size() == rhs.block_size()) && (lhs.min_quantized_numel() == rhs.min_quantized_numel()) &&
           (lhs.bf16_stochastic_round() == rhs.bf16_stochastic_round()) &&
           (lhs.rescale_every() == rhs.rescale_every()) && (lhs.overflow_factor() == rhs.overflow_factor());
}

void FusedAdamW8bitOptions::serialize(torch::serialize::OutputArchive& archive) const 
{
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(betas);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(amsgrad);

    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(block_size);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(min_quantized_numel);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(bf16_stochastic_round);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(rescale_every);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(overflow_factor);
}

void FusedAdamW8bitOptions::serialize(torch::serialize::InputArchive& archive) 
{
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(betas_t, betas);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, amsgrad);

    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, block_size);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, min_quantized_numel);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, bf16_stochastic_round);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, rescale_every);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, overflow_factor);
}

double FusedAdamW8bitOptions::get_lr() const 
{
    return lr();
}

void FusedAdamW8bitOptions::set_lr(const double lr) 
{
    this->lr(lr);
}

FusedAdamW8bit::FusedAdamW8bit(const std::vector<torch::Tensor>& params, FusedAdamW8bitOptions defaults) :
    FusedAdamW8bit({torch::optim::OptimizerParamGroup(std::move(params))}, std::move(defaults)) 
{}

FusedAdamW8bit::FusedAdamW8bit(std::vector<torch::optim::OptimizerParamGroup> param_groups, FusedAdamW8bitOptions defaults) :
    torch::optim::Optimizer(param_groups, std::make_unique<FusedAdamW8bitOptions>(defaults)) 
{
    validate_options(defaults, this->param_groups());
    qmap_signed_cpu_ = create_dynamic_map(true);
    qmap_unsigned_cpu_ = create_dynamic_map(false);
}

void FusedAdamW8bit::validate_options(
    const FusedAdamW8bitOptions& options,
    const std::vector<torch::optim::OptimizerParamGroup>& param_groups
) 
{
    if (options.lr() < 0.0)
    {
        throw std::invalid_argument("Invalid learning rate.");
    }
    if (options.eps() < 0.0)
    {
        throw std::invalid_argument("Invalid epsilon value.");
    }
    if (options.weight_decay() < 0.0) 
    {
        throw std::invalid_argument("Invalid weight_decay value.");
    }
    if (std::get<0>(options.betas()) < 0.0 || std::get<0>(options.betas()) >= 1.0 || 
        std::get<1>(options.betas()) < 0.0 || std::get<1>(options.betas()) >= 1.0) 
    {
        throw std::invalid_argument("Invalid beta values.");
    }
    if (options.block_size() != 256) 
    {
        throw std::invalid_argument("FusedAdamW8bit currently supports block_size=256.");
    }
    if (options.min_quantized_numel() < 1) 
    {
        throw std::invalid_argument("min_quantized_numel must be >= 1.");
    }
    if (options.rescale_every() < 0)
    {
        throw std::invalid_argument("rescale_every must be >= 0.");
    }
    if (options.overflow_factor() < 1.0) 
    {
        throw std::invalid_argument("overflow_factor must be >= 1.0.");
    }
    if (options.amsgrad())
    {
        throw std::invalid_argument("FusedAdamW8bit fast path does not support amsgrad=true.");
    }
    if (param_groups.empty()) 
    {
        throw std::invalid_argument("FusedAdamW8bit requires at least one parameter tensor.");
    }

    bool any_param = false;
    for (const auto& group : param_groups) 
    {
        for (const auto& p : group.params()) 
        {
            any_param = true;
            if (!p.defined()) 
            {
                throw std::invalid_argument("FusedAdamW8bit received an undefined parameter tensor.");
            }
            if (!p.is_leaf()) 
            {
                throw std::invalid_argument("FusedAdamW8bit expects leaf tensors as parameters.");
            }
            if (!p.is_floating_point()) 
            {
                throw std::invalid_argument("FusedAdamW8bit expects floating-point parameter tensors.");
            }
        }
    }
    if (!any_param) 
    {
        throw std::invalid_argument("FusedAdamW8bit requires at least one parameter tensor.");
    }
}

torch::Tensor FusedAdamW8bit::step(torch::optim::Optimizer::LossClosure closure) 
{
    torch::Tensor loss;
    if (closure) 
    {
        torch::AutoGradMode enable_grad(true);
        loss = closure();
    }
    torch::NoGradGuard no_grad;

    auto& opt = this->options();

    const double beta1 = std::get<0>(opt.betas());
    const double beta2 = std::get<1>(opt.betas());
    const double lr = opt.lr();
    const double eps = opt.eps();
    const double weight_decay = opt.weight_decay();

    for (auto& group : this->param_groups()) 
    {        
        for (auto& p : group.params())
        {
            auto grad = p.grad();
            if (!grad.defined()) 
            {
                continue;
            }
            if (grad.is_sparse()) 
            {
                throw std::invalid_argument("FusedAdamW8bit does not support sparse gradients.");
            }
            if (!grad.is_floating_point()) 
            {
                throw std::invalid_argument("FusedAdamW8bit expects floating-point gradients.");
            }
            if (p.device() != grad.device())
            {
                throw std::invalid_argument("Parameter and gradient must be on the same device.");
            }

            auto& state = get_or_init_state(p);
            state.step += 1;

            const bool try_fused = state.quantized && p.is_cuda();
            if (try_fused) 
            {
                auto grad_kernel = grad;
                if (grad_kernel.scalar_type() != p.scalar_type()) 
                {
                    grad_kernel = grad_kernel.to(p.scalar_type());
                }

                const auto qmap_signed = qmap_signed_cpu_.to(p.device(), torch::kFloat32);
                const auto qmap_unsigned = qmap_unsigned_cpu_.to(p.device(), torch::kFloat32);

                try 
                {
                    adamw8bit_fused_cuda_step(
                        p,
                        grad_kernel,
                        state.exp_avg_q.codes,
                        state.exp_avg_sq_q.codes,
                        state.exp_avg_q.absmax,
                        state.exp_avg_sq_q.absmax,
                        qmap_signed,
                        qmap_unsigned,
                        state.step,
                        beta1,
                        beta2,
                        lr,
                        eps,
                        weight_decay,
                        1.0,
                        opt.block_size(),
                        opt.rescale_every(),
                        opt.overflow_factor()
                    );
                    continue;
                } 
                catch (const std::exception&) 
                {
                    // Fall through to the functional fallback if fused CUDA preconditions are not met.
                }
            }

            const auto grad_f32 = grad.contiguous().to(torch::kFloat32);
            auto p_f32 = p.contiguous().to(torch::kFloat32);

            if (weight_decay != 0.0) 
            {
                p_f32 = p_f32 - (lr * weight_decay) * p_f32;
            }

            torch::Tensor exp_avg_f32;
            torch::Tensor exp_avg_sq_f32;
            if (state.quantized) 
            {
                const auto qmap_signed = qmap_signed_cpu_.to(p.device(), torch::kFloat32);
                const auto qmap_unsigned = qmap_unsigned_cpu_.to(p.device(), torch::kFloat32);
                exp_avg_f32 = dequant_to_fp32(state.exp_avg_q, qmap_signed, opt.block_size());
                exp_avg_sq_f32 = dequant_to_fp32(state.exp_avg_sq_q, qmap_unsigned, opt.block_size());
            } 
            else 
            {
                exp_avg_f32 = state.exp_avg_fp32;
                exp_avg_sq_f32 = state.exp_avg_sq_fp32;
            }

            exp_avg_f32 = exp_avg_f32.lerp(grad_f32, 1.0 - beta1);
            exp_avg_sq_f32 = exp_avg_sq_f32.lerp(grad_f32.square(), 1.0 - beta2);

            if (state.quantized) 
            {
                const auto qmap_signed = qmap_signed_cpu_.to(p.device(), torch::kFloat32);
                const auto qmap_unsigned = qmap_unsigned_cpu_.to(p.device(), torch::kFloat32);
                state.exp_avg_q = quantize_from_fp32(exp_avg_f32, qmap_signed, opt.block_size());
                state.exp_avg_sq_q = quantize_from_fp32(exp_avg_sq_f32, qmap_unsigned, opt.block_size());
            } 
            else 
            {
                state.exp_avg_fp32 = exp_avg_f32;
                state.exp_avg_sq_fp32 = exp_avg_sq_f32;
            }

            const double step_d = static_cast<double>(state.step);
            const double bias_correction1 = 1.0 - std::pow(beta1, step_d);
            const double bias_correction2 = 1.0 - std::pow(beta2, step_d);
            auto denom = (exp_avg_sq_f32.sqrt() / std::sqrt(bias_correction2)).add(eps);
            auto new_p_f32 = p_f32 - lr * (exp_avg_f32 / bias_correction1) / denom;

            if (opt.bf16_stochastic_round() && p.scalar_type() == torch::kBFloat16) 
            {
                p.copy_(new_p_f32.to(torch::kBFloat16));
            } 
            else 
            {
                p.copy_(new_p_f32.to(p.scalar_type()));
            }
        }
    }

    return loss;
}

torch::Tensor FusedAdamW8bit::create_dynamic_map(bool signed_map, int max_exponent_bits, int total_bits)
{
    std::vector<float> data;
    data.reserve(static_cast<size_t>(1 << total_bits));

    const int non_sign_bits = total_bits - 1;
    const int additional_items = (1 << (non_sign_bits - max_exponent_bits)) - 1;
    int last_i = 0;

    for (int i = 0; i < max_exponent_bits; ++i)
    {
        last_i = i;
        const int fraction_items =
            signed_map ? ((1 << (i + non_sign_bits - max_exponent_bits)) + 1)
                       : ((1 << (i + non_sign_bits - max_exponent_bits + 1)) + 1);

        auto boundaries = torch::linspace(0.1, 1.0, fraction_items, torch::TensorOptions().dtype(torch::kFloat32));
        auto means = (boundaries.index({Slice(torch::indexing::None, -1)}) + boundaries.index({Slice(1, torch::indexing::None)})) / 2.0;
        const double scale = std::pow(10.0, -(max_exponent_bits - 1) + i);
        auto vals = (means * scale).contiguous().cpu();

        const auto* ptr = vals.data_ptr<float>();
        for (int64_t k = 0; k < vals.numel(); ++k) 
        {
            data.push_back(ptr[k]);
        }
        if (signed_map) 
        {
            for (int64_t k = 0; k < vals.numel(); ++k) 
            {
                data.push_back(-ptr[k]);
            }
        }
    }

    if (additional_items > 0)
    {
        auto boundaries = torch::linspace(0.1, 1.0, additional_items + 1, torch::TensorOptions().dtype(torch::kFloat32));
        auto means = (boundaries.index({Slice(torch::indexing::None, -1)}) + boundaries.index({Slice(1, torch::indexing::None)})) / 2.0;
        const double scale = std::pow(10.0, -(max_exponent_bits - 1) + last_i);
        auto vals = (means * scale).contiguous().cpu();

        const auto* ptr = vals.data_ptr<float>();
        for (int64_t k = 0; k < vals.numel(); ++k)
        {
            data.push_back(ptr[k]);
        }
        if (signed_map)
        {
            for (int64_t k = 0; k < vals.numel(); ++k) 
            {
                data.push_back(-ptr[k]);
            }
        }
    }

    data.push_back(0.0f);
    data.push_back(1.0f);
    std::sort(data.begin(), data.end());

    if (static_cast<int64_t>(data.size()) != (1LL << total_bits)) 
    {
        throw std::runtime_error("create_dynamic_map produced unexpected table size.");
    }

    return torch::tensor(data, torch::TensorOptions().dtype(torch::kFloat32));
}

std::pair<torch::Tensor, torch::Tensor> FusedAdamW8bit::scale_tensor(const torch::Tensor& input, int64_t block_size)
{
    const auto numel = input.numel();
    if (numel % block_size != 0) 
    {
        throw std::invalid_argument("scale_tensor expects input.numel() divisible by block_size.");
    }
    auto in = input.contiguous().to(torch::kFloat32);
    auto reshaped = in.view({-1, block_size});
    auto absmax = std::get<0>(reshaped.abs().max(-1, false)).clamp_min(1e-12);
    auto scaled = reshaped / absmax.view({-1, 1});
    return {scaled.view(input.sizes()).contiguous(), absmax.contiguous()};
}

torch::Tensor FusedAdamW8bit::quantize_8bit_with_qmap(const torch::Tensor& input, const torch::Tensor& qmap)
{
    auto in = input.contiguous().to(torch::kFloat32);
    auto map = qmap.to(in.device(), torch::kFloat32);

    //first for 128    
    auto thresh = map.index({ 128 });
    auto mask = in >= thresh;
    auto codes = mask.to(torch::kLong) * 128;

    //const int bits[] = {64, 32, 16, 8, 4, 2, 1};
    //for (int bit : bits) 
    for (int bit = 64; bit >= 2; bit /= 2)
    {
        thresh = map.index({ codes + bit });

        //map.index(idx) - what does it do:
        //map = [10, 20, 30, 40, 50, 60]
        //idx = [2, 4, 1]
        // => result = [30, 50, 20]

        mask = in >= thresh;

        codes += mask.to(codes.dtype()) * bit;
    }

    //last for 1
    mask = in >= thresh;
    thresh = map.index({ codes + 1 });
    codes += mask.to(codes.dtype());

    //in codes, there is index of found value

    //rounding
    auto codes_up = (codes + 1).clamp_max(255);

    auto val_down = map.index({ codes });
    auto val_up = map.index({ codes_up });
    auto residual = in - val_down;

    codes = torch::where(residual >= (val_up - val_down) * 0.5, codes_up, codes);

    return codes.to(torch::kUInt8).contiguous();
}

torch::Tensor FusedAdamW8bit::dequant_with_qmap(
    const torch::Tensor& codes,
    const torch::Tensor& qmap,
    const torch::Tensor& absmax,
    int64_t block_size
) 
{
    auto flat_codes = codes.contiguous().view({-1});
    auto map = qmap.to(codes.device(), torch::kFloat32);
    auto deq = map.index({flat_codes.to(torch::kLong)});

    auto block_idx =
        torch::arange(flat_codes.numel(), torch::TensorOptions().dtype(torch::kLong).device(codes.device())) / block_size;
    auto scales = absmax.index({block_idx.to(absmax.device())}).to(torch::kFloat32).to(codes.device());
    auto out = deq * scales;
    return out.view(codes.sizes()).contiguous();
}

FusedAdamW8bit::QuantizedState FusedAdamW8bit::quantize_from_fp32(
    const torch::Tensor& fp32,
    const torch::Tensor& qmap,
    int64_t block_size
) 
{
    auto [scaled, absmax] = scale_tensor(fp32, block_size);
    auto codes = quantize_8bit_with_qmap(scaled, qmap);
    return {codes, absmax};
}

torch::Tensor FusedAdamW8bit::dequant_to_fp32(const QuantizedState& qstate, const torch::Tensor& qmap, int64_t block_size) 
{
    return dequant_with_qmap(qstate.codes, qmap, qstate.absmax, block_size).to(torch::kFloat32);
}

FusedAdamW8bit::QuantizedState FusedAdamW8bit::new_quantized_state(const torch::Tensor& param) const 
{
    const auto block_size = this->options().block_size();
    const int64_t blocks = (param.numel() + block_size - 1) / block_size;

    auto codes = torch::zeros(param.sizes(), param.options().dtype(torch::kUInt8));
    auto absmax = torch::zeros({blocks}, param.options().dtype(torch::kFloat32));
    return {codes, absmax};
}

FusedAdamW8bit::ParamState& FusedAdamW8bit::get_or_init_state(const torch::Tensor& param) 
{
    auto* key = static_cast<void*>(param.unsafeGetTensorImpl());
    auto it = state_.find(key);
    if (it != state_.end()) 
    {
        return it->second;
    }

    auto& opt = this->options();

    ParamState s;
    s.quantized = (param.numel() >= opt.min_quantized_numel()) && (param.numel() % opt.block_size() == 0);

    if (s.quantized) 
    {
        s.exp_avg_q = new_quantized_state(param);
        s.exp_avg_sq_q = new_quantized_state(param);
    } 
    else 
    {
        auto fp32_opts = param.options().dtype(torch::kFloat32);
        s.exp_avg_fp32 = torch::zeros(param.sizes(), fp32_opts);
        s.exp_avg_sq_fp32 = torch::zeros(param.sizes(), fp32_opts);
    }

    auto [inserted_it, inserted] = state_.emplace(key, std::move(s));
    (void)inserted;
    return inserted_it->second;
}
