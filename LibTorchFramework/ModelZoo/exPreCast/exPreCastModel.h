#ifndef EXPRECAST_MODEL_H
#define EXPRECAST_MODEL_H

class OpticalFlowBase;

#include <vector>
#include <array>
#include <memory>

#include <torch/torch.h>

#include "../../core/AbstractModel.h"

#include "./PatchEmbed3D.h"
#include "./BasicLayerSkip.h"
#include "./CubicDualUpsample.h"
#include "./PatchExpanding3D.h"

/// <summary>
/// ChatGPT 
/// </summary>
class DetailAwareFusionImpl : public torch::nn::Module
{
public:
    DetailAwareFusionImpl(int64_t channels, int64_t hiddenChannels = 64)
    {
        encoder = register_module("encoder", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels * 4, hiddenChannels, 3).padding(1)),
            torch::nn::GELU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(hiddenChannels, hiddenChannels, 3).padding(1)),
            torch::nn::GELU()
        ));

        gate = register_module("gate", torch::nn::Conv2d(torch::nn::Conv2dOptions(hiddenChannels, channels, 1)));
        correction = register_module("correction", torch::nn::Conv2d(torch::nn::Conv2dOptions(hiddenChannels, channels, 3).padding(1)));

        // Initial behavior:
        // alpha = 0.5
        // correction = 0
        torch::NoGradGuard noGrad;

        gate->weight.zero_();
        gate->bias.zero_();

        correction->weight.zero_();
        correction->bias.zero_();
    }

    torch::Tensor forward(const torch::Tensor& unet, const torch::Tensor& flow)
    {
        // Input: [B, SeqLen, C, H, W]

        const int64_t B = unet.size(0);
        const int64_t S = unet.size(1);
        const int64_t C = unet.size(2);
        const int64_t H = unet.size(3);
        const int64_t W = unet.size(4);

        auto u = unet.reshape({ B * S, C, H, W });
        auto f = flow.reshape({ B * S, C, H, W });

        auto flowDetail = HighPass(f);

        auto x = torch::cat(
            {
                u,
                f,
                torch::abs(u - f),
                flowDetail
            }, 1);

        auto features = encoder->forward(x);

        auto alpha = torch::sigmoid(gate->forward(features));

        auto out = alpha * u + (1.0f - alpha) * f;
        out += correction->forward(features);

        return out.reshape({ B, S, C, H, W });
    }

private:
    static torch::Tensor HighPass(const torch::Tensor& x)
    {
        auto low = torch::avg_pool2d(
            x,
            { 3, 3 },
            { 1, 1 },
            { 1, 1 }
        );

        return x - low;
    }

private:
    torch::nn::Sequential encoder{ nullptr };
    torch::nn::Conv2d gate{ nullptr };
    torch::nn::Conv2d correction{ nullptr };
};

TORCH_MODULE(DetailAwareFusion);

namespace ModelZoo {
    namespace exPreCast {
        class exPreCastModel : public AbstractModel
        {
        public:
            exPreCastModel(
                int64_t inputFrames = 12,
                int64_t outputFrames = 12,
                int64_t inChans = 1,
                int64_t outChans = 1,
                std::array<int64_t, 3> patchEmbedSize = { 2, 4, 4 },
                std::array<int64_t, 3> patchExpandSize = { 2, 4, 4 },
                std::array<int64_t, 3> upsamplingScale = { 1, 2, 2 },
                std::array<int64_t, 3> downsamplingScale = { 1, 2, 2 },
                int64_t embedDim = 96,
                std::vector<int64_t> depths = { 2, 6, 2, 2 },
                std::vector<int64_t> numHeads = { 3, 6, 12, 24 },
                std::array<int64_t, 3> windowSize = { 2, 7, 7 },
                double mlpRatio = 4.0,
                bool qkvBias = true,
                double dropRate = 0.0,
                double attnDropRate = 0.0,
                double dropPathRate = 0.2,
                bool patchNorm = false,                
                std::string skipConnection = "add"
            );

            const char* GetName() const override;

            torch::Tensor forward(torch::Tensor x);
            
            std::vector<torch::Tensor> RunForward(DataLoaderData& batch) override;

        private:
                  
            int64_t inputFrames;
            int64_t outputFrames;

            int64_t inChans;
            int64_t outChans;

            int64_t embedDim;
            int64_t numLayers;

            std::array<int64_t, 3> patchEmbedSize;
            std::array<int64_t, 3> patchExpandSize;

            std::array<int64_t, 3> upsamplingScale;
            std::array<int64_t, 3> downsamplingScale;

            std::string skipConnection;

            int64_t lastTimeDim;

            PatchEmbed3D patchEmbed{ nullptr };
            torch::nn::Dropout posDrop{ nullptr };

            torch::nn::ModuleList encoder;
            torch::nn::ModuleList decoder;

            CubicDualUpsample bottleneckUpscale{ nullptr };

            PatchExpanding3D patchExpand3D{ nullptr };

            torch::nn::Conv3d timeExtractor{ nullptr };       

            std::shared_ptr<OpticalFlowBase> flow;

            DetailAwareFusion fusion{ nullptr };
        };
    }
}


#endif