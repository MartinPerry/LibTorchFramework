#pragma once

#include <optional>
#include <vector>
#include <array>

#include <torch/torch.h>

#include "SwinTransformerBlock3D.h"
#include "WindowUtils.h"

class BasicLayerSkipImpl : public torch::nn::Module
{
public:
    enum SubsampleType
    {
        PatchMergingType,
        CubicDualUpsampleType,
        None
    };

    BasicLayerSkipImpl(
        int64_t dim,
        int64_t depth,
        int64_t numHeads,
        const std::array<int64_t, 3>& windowSize = { 1, 7, 7 },
        double mlpRatio = 4.0,
        bool qkvBias = false,
        std::optional<double> qkScale = std::nullopt,
        double drop = 0.0,
        double attnDrop = 0.0,
        std::vector<double> dropPaths = {0.0},
        SubsampleType subsampleType = SubsampleType::None,
        const std::array<int64_t, 3>& subsampleScale = {1, 2, 2}
    );

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

private:
    torch::Tensor computeMask(
        int64_t D,
        int64_t H,
        int64_t W,
        const std::array<int64_t, 3>& windowSize,
        const std::array<int64_t, 3>& shiftSize,
        torch::Device device);

private:
    std::array<int64_t, 3> windowSize;
    std::array<int64_t, 3> shiftSize;
    int64_t depth;

    torch::nn::ModuleList blocks;

    //PatchMerging subsample{ nullptr };
    torch::nn::AnyModule subsample;
};

TORCH_MODULE(BasicLayerSkip);