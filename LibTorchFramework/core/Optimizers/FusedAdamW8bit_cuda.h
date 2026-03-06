#pragma once

#include <cstdint>

#include <torch/torch.h>

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
);
