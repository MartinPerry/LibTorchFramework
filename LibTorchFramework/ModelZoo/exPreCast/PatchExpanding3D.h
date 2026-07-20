#pragma once

#include <torch/torch.h>
#include <array>

class PatchExpanding3DImpl : public torch::nn::Module
{
public:
    PatchExpanding3DImpl(
        const std::array<int64_t, 3>& patchSize = { 2, 4, 4 },
        int64_t embedDim = 96,
        int64_t outChans = 3);

    torch::Tensor forward(torch::Tensor x);

private:
    std::array<int64_t, 3> patchSize;
    int64_t embedDim;
    int64_t outChans;

    torch::nn::ConvTranspose3d deproj{ nullptr };
};

TORCH_MODULE(PatchExpanding3D);