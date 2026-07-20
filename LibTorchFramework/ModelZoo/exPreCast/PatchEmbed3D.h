#pragma once

#include <torch/torch.h>
#include <array>

class PatchEmbed3DImpl : public torch::nn::Module
{
public:
    PatchEmbed3DImpl(
        const std::array<int64_t, 3>& patchSize = { 2, 4, 4 },
        int64_t inChans = 3,
        int64_t embedDim = 96,
        bool useNorm = false);

    torch::Tensor forward(torch::Tensor x);

private:
    std::array<int64_t, 3> patchSize;

    int64_t inChans;
    int64_t embedDim;

    torch::nn::Conv3d proj{ nullptr };
    torch::nn::LayerNorm norm{ nullptr };
};

TORCH_MODULE(PatchEmbed3D);