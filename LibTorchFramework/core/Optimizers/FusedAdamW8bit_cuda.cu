#include "./FusedAdamW8bit_cuda.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int kBlockSize = 256;

__device__ inline uint8_t quantize_code_nearest(float x, const float* qmap) 
{
    // Saturate to the representable range before branchless binary search.
    x = fminf(fmaxf(x, qmap[0]), qmap[255]);

    int code = (x >= qmap[128]) ? 128 : 0;
#pragma unroll
    for (int bit = 64; bit >= 1; bit >>= 1) 
    {
        const int candidate = min(code + bit, 255);
        if (x >= qmap[candidate]) 
        {
            code = candidate;
        }
    }

    const int up = min(code + 1, 255);
    const float down_v = qmap[code];
    const float up_v = qmap[up];
    if (fabsf(x - up_v) < fabsf(x - down_v))
    {
        code = up;
    }
    return static_cast<uint8_t>(code);
}

template <typename scalar_t>
__global__ void adamw8bit_fused_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    uint8_t* __restrict__ m_codes,
    uint8_t* __restrict__ v_codes,
    float* __restrict__ m_absmax,
    float* __restrict__ v_absmax,
    const float* __restrict__ qmap_signed,
    const float* __restrict__ qmap_unsigned,
    int64_t n,
    float beta1,
    float beta2,
    float lr,
    float eps,
    float weight_decay,
    float gnorm_scale,
    float correction1,
    float correction2,
    int periodic_rescale,
    float overflow_factor
) 
{
    const int block = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t idx = static_cast<int64_t>(block) * kBlockSize + tid;

    __shared__ float sm_qmap_signed[256];
    __shared__ float sm_qmap_unsigned[256];
    __shared__ int sm_need_rescale;
    __shared__ float sm_m_abs[kBlockSize];
    __shared__ float sm_v_abs[kBlockSize];
    __shared__ float sm_new_m_scale;
    __shared__ float sm_new_v_scale;

    sm_qmap_signed[tid] = qmap_signed[tid];
    sm_qmap_unsigned[tid] = qmap_unsigned[tid];
    if (tid == 0) 
    {
        sm_need_rescale = periodic_rescale;
    }
    __syncthreads();

    const float old_m_scale = fmaxf(m_absmax[block], 1e-12f);
    const float old_v_scale = fmaxf(v_absmax[block], 1e-12f);

    float local_m = 0.0f;
    float local_v = 0.0f;
    float local_p = 0.0f;

    if (idx < n) 
    {
        local_p = static_cast<float>(param[idx]);
        float g = static_cast<float>(grad[idx]);

        const uint8_t m_code = m_codes[idx];
        const uint8_t v_code = v_codes[idx];

        local_m = sm_qmap_signed[m_code] * old_m_scale;
        local_v = sm_qmap_unsigned[v_code] * old_v_scale;

        if (::isfinite(g)) 
        {
            g *= gnorm_scale;
            local_m = beta1 * local_m + (1.0f - beta1) * g;
            local_v = beta2 * local_v + (1.0f - beta2) * g * g;

            const float m_hat = local_m / correction1;
            const float v_hat_sqrt = sqrtf(local_v / correction2);
            local_p = local_p - lr * (m_hat / (v_hat_sqrt + eps));
            if (weight_decay != 0.0f)
            {
                local_p *= (1.0f - lr * weight_decay);
            }
        }
    }

    float m_abs = (idx < n) ? fabsf(local_m) : 0.0f;
    float v_abs = (idx < n) ? fabsf(local_v) : 0.0f;
    sm_m_abs[tid] = m_abs;
    sm_v_abs[tid] = v_abs;

    if (!periodic_rescale)
    {
        const bool overflow = (m_abs > old_m_scale * overflow_factor) || (v_abs > old_v_scale * overflow_factor);
        if (overflow)
        {
            atomicExch(&sm_need_rescale, 1);
        }
    }
    __syncthreads();

    float new_m_scale = old_m_scale;
    float new_v_scale = old_v_scale;

    if (sm_need_rescale) 
    {
        for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) 
        {
            if (tid < stride) 
            {
                sm_m_abs[tid] = fmaxf(sm_m_abs[tid], sm_m_abs[tid + stride]);
                sm_v_abs[tid] = fmaxf(sm_v_abs[tid], sm_v_abs[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) 
        {
            sm_new_m_scale = fmaxf(sm_m_abs[0], 1e-12f);
            sm_new_v_scale = fmaxf(sm_v_abs[0], 1e-12f);
            m_absmax[block] = sm_new_m_scale;
            v_absmax[block] = sm_new_v_scale;
        }
        __syncthreads();
        new_m_scale = sm_new_m_scale;
        new_v_scale = sm_new_v_scale;
    }

    if (idx < n) 
    {
        const float m_norm = local_m / new_m_scale;
        const float v_norm = local_v / new_v_scale;

        m_codes[idx] = quantize_code_nearest(m_norm, sm_qmap_signed);
        v_codes[idx] = quantize_code_nearest(v_norm, sm_qmap_unsigned);
        param[idx] = static_cast<scalar_t>(local_p);
    }
}

} // namespace

void adamw8bit_fused_cuda_step(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor m_codes,
    torch::Tensor v_codes,
    torch::Tensor m_absmax,
    torch::Tensor v_absmax,
    torch::Tensor qmap_signed,
    torch::Tensor qmap_unsigned,
    int64_t step,
    double beta1,
    double beta2,
    double lr,
    double eps,
    double weight_decay,
    double gnorm_scale,
    int64_t block_size,
    int64_t rescale_every,
    double overflow_factor
) 
{
    if (!param.is_cuda())
    {
        throw std::invalid_argument("adamw8bit_fused_cuda_step expects CUDA tensors.");
    }
    if (!grad.is_cuda() || !m_codes.is_cuda() || !v_codes.is_cuda() || !m_absmax.is_cuda() || !v_absmax.is_cuda()) 
    {
        throw std::invalid_argument("adamw8bit_fused_cuda_step expects all state tensors on CUDA.");
    }
    if (!qmap_signed.is_cuda() || !qmap_unsigned.is_cuda()) 
    {
        throw std::invalid_argument("adamw8bit_fused_cuda_step expects qmaps on CUDA.");
    }
    if (param.numel() != grad.numel() || param.numel() != m_codes.numel() || param.numel() != v_codes.numel()) 
    {
        throw std::invalid_argument("adamw8bit_fused_cuda_step: tensor numel mismatch.");
    }
    if (m_codes.scalar_type() != torch::kUInt8 || v_codes.scalar_type() != torch::kUInt8)
    {
        throw std::invalid_argument("adamw8bit_fused_cuda_step expects uint8 moment codes.");
    }
    if (m_absmax.scalar_type() != torch::kFloat32 || v_absmax.scalar_type() != torch::kFloat32 ||
        qmap_signed.scalar_type() != torch::kFloat32 || qmap_unsigned.scalar_type() != torch::kFloat32) 
    {
        throw std::invalid_argument("adamw8bit_fused_cuda_step expects float32 absmax/qmaps.");
    }
    if (qmap_signed.numel() != 256 || qmap_unsigned.numel() != 256) 
    {
        throw std::invalid_argument("adamw8bit_fused_cuda_step: qmaps must have 256 entries.");
    }
    if (block_size != kBlockSize) 
    {
        throw std::invalid_argument("adamw8bit_fused_cuda_step currently supports block_size=256 only.");
    }
    if (step < 1)
    {
        throw std::invalid_argument("adamw8bit_fused_cuda_step expects step >= 1.");
    }

    const auto n = param.numel();
    if (n == 0) 
    {
        return;
    }

    const int64_t blocks = (n + kBlockSize - 1) / kBlockSize;
    if (m_absmax.numel() != blocks || v_absmax.numel() != blocks) 
    {
        throw std::invalid_argument("adamw8bit_fused_cuda_step: absmax tensor size mismatch.");
    }

    auto p = param.contiguous();
    auto g = grad.contiguous();
    auto m = m_codes.contiguous();
    auto v = v_codes.contiguous();
    auto m_scale = m_absmax.contiguous();
    auto v_scale = v_absmax.contiguous();
    auto q1 = qmap_signed.contiguous();
    auto q2 = qmap_unsigned.contiguous();

    // Keep the caller's storage updates visible even when contiguous() returns aliases.
    if (!p.is_alias_of(param) || !m.is_alias_of(m_codes) || !v.is_alias_of(v_codes) || 
        !m_scale.is_alias_of(m_absmax) || !v_scale.is_alias_of(v_absmax)) 
    {
        throw std::invalid_argument("adamw8bit_fused_cuda_step expects contiguous tensors.");
    }

    const float beta1_f = static_cast<float>(beta1);
    const float beta2_f = static_cast<float>(beta2);
    const float lr_f = static_cast<float>(lr);
    const float eps_f = static_cast<float>(eps);
    const float wd_f = static_cast<float>(weight_decay);
    const float gnorm_f = static_cast<float>(gnorm_scale);
    const float overflow_f = static_cast<float>(overflow_factor);

    const float correction1 = 1.0f - std::pow(beta1_f, static_cast<float>(step));
    const float correction2 = 1.0f - std::pow(beta2_f, static_cast<float>(step));
    const int periodic_rescale = (rescale_every > 0 && (step % rescale_every == 0)) ? 1 : 0;

    c10::cuda::CUDAGuard device_guard(param.device());
    const auto stream = at::cuda::getCurrentCUDAStream(param.get_device()).stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kHalf,
        torch::kBFloat16,
        p.scalar_type(),
        "adamw8bit_fused_kernel",
        [&] {
            adamw8bit_fused_kernel<scalar_t><<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                p.data_ptr<scalar_t>(),
                g.data_ptr<scalar_t>(),
                m.data_ptr<uint8_t>(),
                v.data_ptr<uint8_t>(),
                m_scale.data_ptr<float>(),
                v_scale.data_ptr<float>(),
                q1.data_ptr<float>(),
                q2.data_ptr<float>(),
                n,
                beta1_f,
                beta2_f,
                lr_f,
                eps_f,
                wd_f,
                gnorm_f,
                correction1,
                correction2,
                periodic_rescale,
                overflow_f
            );
        }
    );
}
