#pragma once

#include <torch/torch.h>


class DetailAwareFusionImpl : public torch::nn::Module
{
public:
    DetailAwareFusionImpl(
        int64_t channels,
        int64_t hiddenChannels = 64,
        float maxCorrection = 0.10f);

    torch::Tensor forward(
        const torch::Tensor& unet,
        const torch::Tensor& flow);

private:
    static torch::Tensor HighPass(
        const torch::Tensor& x);

    static torch::Tensor LowPass(
        const torch::Tensor& x);

private:
    int64_t channels;
    int64_t hiddenChannels;
    float maxCorrection;

    torch::nn::Sequential encoder{ nullptr };

    torch::nn::Conv2d gate{ nullptr };
    torch::nn::Conv2d correction{ nullptr };
};


TORCH_MODULE(DetailAwareFusion);

