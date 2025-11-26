#include "./encoder.h"

#include "./decoder.h"

using namespace ModelZoo::sdvae;

VAE_EncoderImpl::VAE_EncoderImpl()
{
    seq = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 128, 3).padding(1)),

        VAE_ResidualBlock(128, 128),
        VAE_ResidualBlock(128, 128),

        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(2)),

        VAE_ResidualBlock(128, 256),
        VAE_ResidualBlock(256, 256),

        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(2)),

        VAE_ResidualBlock(256, 512),
        VAE_ResidualBlock(512, 512),

        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(2)),

        VAE_ResidualBlock(512, 512),
        VAE_ResidualBlock(512, 512),
        VAE_ResidualBlock(512, 512),

        VAE_AttentionBlock(512),

        VAE_ResidualBlock(512, 512),

        torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, 512)),
        torch::nn::SiLU(),

        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 8, 3).padding(1)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 8, 1))
    );

    seq = register_module("seq", seq);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> VAE_EncoderImpl::forward(
    torch::Tensor x, torch::Tensor noise)
{
    for (auto& layer : *seq)
    {
        // detect Conv2d with stride==2        
        auto conv = std::dynamic_pointer_cast<torch::nn::Conv2d>(layer.ptr());        
        if (conv != nullptr)
        {            
            if (conv->get()->options.stride()->at(0) == 2 && conv->get()->options.stride()->at(1) == 2)
            {
                // asymmetric padding: (left, right, top, bottom)
                x = torch::constant_pad_nd(x, { 0, 1, 0, 1 }, 0);
            }
        }

        x = layer.forward(x);
    }

    // Split into mean/logvar
    auto chunks = x.chunk(2, 1);
    torch::Tensor mean = chunks[0];
    torch::Tensor log_variance = chunks[1];

    log_variance = torch::clamp(log_variance, -30.0, 20.0);
    torch::Tensor variance = torch::exp(log_variance);
    torch::Tensor stdev = torch::sqrt(variance);

    if (!noise.defined())
    {
        noise = torch::randn_like(stdev);
    }

    x = mean + stdev * noise;

    x = x * 0.18215;

    return { x, mean, log_variance };
}

torch::Tensor VAE_EncoderImpl::kl_divergence(
    torch::Tensor mean, torch::Tensor log_variance)
{
    torch::Tensor kl = -0.5 * (1 + log_variance - mean.pow(2) - torch::exp(log_variance));

    return kl.sum({ 1, 2 }, true);
}