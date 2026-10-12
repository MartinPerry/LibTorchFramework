
#pragma once

#include <torch/torch.h>

namespace ModelZoo::exPreCast
{
    class OpticalForecastEncoderImpl : public torch::nn::Module
    {
    public:
        explicit OpticalForecastEncoderImpl(
            int64_t inChannels = 1,
            int64_t hiddenChannels = 32,
            int64_t outChannels = 64)
        {
            encoder = register_module(
                "encoder",
                torch::nn::Sequential(
                    // [B, 1, M, H, W] -> [B, 32, M, H/2, W/2]
                    torch::nn::Conv3d(
                        torch::nn::Conv3dOptions(
                            inChannels, hiddenChannels,
                            { 1, 3, 3 }
                        )
                        .stride({ 1, 2, 2 })
                        .padding({ 0, 1, 1 })
                    ),
                    torch::nn::GELU(),

                    torch::nn::Conv3d(
                        torch::nn::Conv3dOptions(
                            hiddenChannels, hiddenChannels,
                            { 3, 3, 3 }
                        )
                        .padding(1)
                    ),
                    torch::nn::GELU(),

                    // [B, 32, M, H/2, W/2]
                    // -> [B, 64, M, H/4, W/4]
                    torch::nn::Conv3d(
                        torch::nn::Conv3dOptions(
                            hiddenChannels, outChannels,
                            { 1, 3, 3 }
                        )
                        .stride({ 1, 2, 2 })
                        .padding({ 0, 1, 1 })
                    ),
                    torch::nn::GELU(),

                    // Learn interactions between adjacent forecast frames.
                    torch::nn::Conv3d(
                        torch::nn::Conv3dOptions(
                            outChannels, outChannels,
                            { 3, 3, 3 }
                        )
                        .padding(1)
                    ),
                    torch::nn::GELU()
                )
            );
        }

        torch::Tensor forward(const torch::Tensor& optical)
        {
            return encoder->forward(optical);
        }

    private:
        torch::nn::Sequential encoder{ nullptr };
    };

    TORCH_MODULE(OpticalForecastEncoder);
}
