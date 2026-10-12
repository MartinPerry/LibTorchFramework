#pragma once

#include <torch/torch.h>

namespace ModelZoo::exPreCast
{
    class MotionConditionedFusionImpl : public torch::nn::Module
    {
    public:
        MotionConditionedFusionImpl(
            int64_t channels,
            int64_t opticalChannels = 64
        )
        {
            opticalProject = register_module(
                "optical_project",
                torch::nn::Conv3d(
                    torch::nn::Conv3dOptions(
                        opticalChannels, channels, 1)
                )
            );

            refine = register_module(
                "refine",
                torch::nn::Sequential(
                    torch::nn::Conv3d(
                        torch::nn::Conv3dOptions(
                            channels * 2, channels, 3
                        ).padding(1)
                    ),
                    torch::nn::GELU(),
                    torch::nn::Conv3d(
                        torch::nn::Conv3dOptions(
                            channels, channels, 3
                        ).padding(1)
                    )
                )
            );

            gate = register_module(
                "gate",
                torch::nn::Conv3d(
                    torch::nn::Conv3dOptions(
                        channels * 2, channels, 1)
                )
            );

            // Start close to the original decoder behavior.
            // Unlike an exactly zero scale, this allows gradients
            // to reach the optical branch immediately.
            alpha = register_parameter(
                "alpha",
                torch::full({}, 0.1f)
            );
        }

        torch::Tensor forward(
            const torch::Tensor& decoder,
            const torch::Tensor& opticalFeatures)
        {
            // decoder:        [B, C, T, H, W]
            // opticalFeatures:[B, Co, M, Ho, Wo]

            auto optical = torch::nn::functional::interpolate(
                opticalFeatures,
                torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{
                decoder.size(2),
                    decoder.size(3),
                    decoder.size(4)
            })
                .mode(torch::kTrilinear)
                .align_corners(false)
            );

            optical = opticalProject->forward(optical);

            auto combined = torch::cat(
                { decoder, optical }, 1
            );

            auto correction = refine->forward(combined);
            auto weights = torch::sigmoid(gate->forward(combined));

            return decoder + alpha * weights * correction;
        }

    private:
        torch::nn::Conv3d opticalProject{ nullptr };
        torch::nn::Sequential refine{ nullptr };
        torch::nn::Conv3d gate{ nullptr };

        torch::Tensor alpha;
    };

    TORCH_MODULE(MotionConditionedFusion);
}