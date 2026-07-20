#pragma once

#include <torch/torch.h>
#include <array>

class WindowAttention3DImpl : public torch::nn::Module
{
public:
    WindowAttention3DImpl(
        int64_t dim,
        const std::array<int64_t, 3>& windowSize,
        int64_t numHeads,
        bool qkvBias = false,
        std::optional<double> qkScale = std::nullopt,
        double attnDrop = 0.0,
        double projDrop = 0.0);

    torch::Tensor forward(
        torch::Tensor x,
        std::optional<torch::Tensor> mask = std::nullopt);

private:
    int64_t dim;
    std::array<int64_t, 3> windowSize;
    int64_t numHeads;
    double scale;

    torch::Tensor relativePositionBiasTable;
    torch::Tensor relativePositionIndex;

    torch::nn::Linear qkv{ nullptr };
    torch::nn::Dropout attnDrop{ nullptr };
    torch::nn::Linear proj{ nullptr };
    torch::nn::Dropout projDrop{ nullptr };
    torch::nn::Softmax softmax{ nullptr };
};

TORCH_MODULE(WindowAttention3D);