#pragma once

#include <torch/torch.h>
#include <array>

#include "PixelShuffle3D.h"

class CubicDualUpsampleImpl : public torch::nn::Module
{
public:
    CubicDualUpsampleImpl(
        int64_t dim,
        const std::array<int64_t, 3>& scale = { 1, 2, 2 },
        int64_t kernelSize = 1,
        int64_t strideSize = 1,
        int64_t padding = 0);

    torch::Tensor forward(torch::Tensor x);

private:
    int64_t dim;
    std::array<int64_t, 3> scale;

    int64_t scaleFactor;

    torch::nn::Conv3d convP1{ nullptr };
    torch::nn::PReLU act{ nullptr };
    PixelShuffle3D pixelShuffle{ nullptr };
    torch::nn::Conv3d convP2{ nullptr };

    torch::nn::Conv3d convB1{ nullptr };
    //torch::nn::Upsample upSample{ nullptr };
    torch::nn::Conv3d convB2{ nullptr };

    torch::nn::Conv3d convMerge{ nullptr };
    torch::nn::LayerNorm norm{ nullptr };
};

TORCH_MODULE(CubicDualUpsample);