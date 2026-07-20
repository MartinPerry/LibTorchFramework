#pragma once

#include <torch/torch.h>

class DropPathImpl : public torch::nn::Module
{
public:
    DropPathImpl(double dropProb = 0.0, bool scaleByKeep = true);

    torch::Tensor forward(torch::Tensor x);

private:
    double dropProb;
    bool scaleByKeep;
};

TORCH_MODULE(DropPath);