#pragma once

#include <torch/torch.h>
#include <array>

#include "../../core/Modules/Convolutions/DeformConv.h"

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
    std::array<int64_t, 3> scale;
    
    //Conv3d
    torch::nn::AnyModule convP1;
    torch::nn::PReLU act{ nullptr };
    PixelShuffle3D pixelShuffle{ nullptr };
    torch::nn::AnyModule convP2;

    torch::nn::AnyModule convB1;
    torch::nn::AnyModule convB2;

    torch::nn::Conv3d convMerge{ nullptr };
    torch::nn::LayerNorm norm{ nullptr };

    torch::nn::Conv3d CreateDefaultConv3d(int64_t inC, int64_t outC, 
        int64_t kernelSize,
        int64_t strideSize,
        int64_t padding,
        bool bias);

    StackDeformConv3d CreateDeformConv3d(int64_t inC, int64_t outC,
        int64_t kernelSize,
        int64_t strideSize,
        int64_t padding,
        bool bias);
};

TORCH_MODULE(CubicDualUpsample);