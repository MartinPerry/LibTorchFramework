#pragma once

#include <optional>
#include <vector>
#include <array>

#include <torch/torch.h>

#include "SwinTransformerBlock3D.h"
#include "PatchMerging.h"
#include "WindowUtils.h"

class BasicLayerSkipImpl : public torch::nn::Module
{
public:
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
        double dropPath = 0.0,
        bool useSubsample = false);

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
    bool useCheckpoint;

    torch::nn::ModuleList blocks;

    PatchMerging subsample{ nullptr };
};

TORCH_MODULE(BasicLayerSkip);