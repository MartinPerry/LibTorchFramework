#pragma once

#include <torch/torch.h>
#include <array>

#include "../../core/Modules/MLP.h"
#include "../../core/Modules/DropPath.h"

#include "./WindowAttention3D.h"


class SwinTransformerBlock3DImpl : public torch::nn::Module
{
public:
    SwinTransformerBlock3DImpl(
        int64_t dim,
        int64_t numHeads,
        const std::array<int64_t, 3>& windowSize = { 2, 7, 7 },
        const std::array<int64_t, 3>& shiftSize = { 0, 0, 0 },
        double mlpRatio = 4.0,
        bool qkvBias = true,
        std::optional<double> qkScale = std::nullopt,
        double drop = 0.0,
        double attnDrop = 0.0,
        double dropPath = 0.0);

    torch::Tensor forward(torch::Tensor x, torch::Tensor maskMatrix);

private:
    torch::Tensor forwardPart1(torch::Tensor x, torch::Tensor maskMatrix);

    torch::Tensor forwardPart2(torch::Tensor x);

private:
    int64_t dim;
    int64_t numHeads;
    std::array<int64_t, 3> windowSize;
    std::array<int64_t, 3> shiftSize;
    double mlpRatio;
    bool useCheckpoint;

    torch::nn::LayerNorm norm1{ nullptr };
    WindowAttention3D attn{ nullptr };
    DropPath dropPathLayer{ nullptr };
    torch::nn::Identity identity;
    torch::nn::LayerNorm norm2{ nullptr };
    Mlp mlp{ nullptr };
};

TORCH_MODULE(SwinTransformerBlock3D);