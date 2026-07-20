#pragma once

#include <torch/torch.h>
#include <array>

class PixelShuffle3DImpl : public torch::nn::Module
{
public:
    PixelShuffle3DImpl(const std::array<int64_t, 3>& scale);

    torch::Tensor forward(torch::Tensor input);

private:
    std::array<int64_t, 3> scale;
};

TORCH_MODULE(PixelShuffle3D);