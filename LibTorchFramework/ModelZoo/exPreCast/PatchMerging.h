#pragma once

#include <torch/torch.h>

class PatchMergingImpl : public torch::nn::Module
{
public:
    PatchMergingImpl(int64_t dim);

    torch::Tensor forward(torch::Tensor x);

private:
    int64_t dim;

    torch::nn::Linear reduction{ nullptr };
    torch::nn::LayerNorm norm{ nullptr };
};

TORCH_MODULE(PatchMerging);