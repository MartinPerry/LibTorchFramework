#ifndef DEFORM_CONV_H
#define DEFORM_CONV_H

#include <utility>
#include <optional>
#include <tuple>
#include <array>

#include <torch/torch.h>

//==================================================================================================
// DeformConv2d
//==================================================================================================

class DeformConv2dImpl : public torch::nn::Module
{
public:

    DeformConv2dImpl(
        int64_t in_channels,
        int64_t out_channels,
        std::array<int64_t, 2> kernelSizes = { 3, 3 },
        std::array<int64_t, 2> strides = { 1, 1 },
        std::array<int64_t, 2> paddings = { 1, 1 },
        std::array<int64_t, 2> dilations = { 1, 1 },
        bool useBias = true,
        bool useMask = false,
        bool useAutoOffset = true
    );

    void reset_parameters();

    torch::Tensor forward(
        torch::Tensor x,
        std::optional<torch::Tensor> baseOffset = std::nullopt,
        std::optional<torch::Tensor> mask = std::nullopt
    );

private:

    torch::Tensor weight;
    torch::Tensor bias;

    torch::nn::Conv2d convDirs{ nullptr };
    torch::nn::Conv2d convOffsetFromX{ nullptr };
    torch::nn::Conv2d maskConv{ nullptr };
    
    int64_t in_channels;
    int64_t out_channels;
    std::array<int64_t, 2> kernelSize;
    std::array<int64_t, 2> stride;
    std::array<int64_t, 2> padding;
    std::array<int64_t, 2> dilation;
    int64_t groups;
    int64_t groupsOffset;

    bool useMask;
};

TORCH_MODULE(DeformConv2d);

//==================================================================================================
// DeformConv3d
//==================================================================================================

class DeformConv3dImpl : public torch::nn::Module
{
public:

    DeformConv3dImpl(
        int64_t in_channels,
        int64_t out_channels,
        std::array<int64_t, 3> kernelSizes = { 3, 3, 3 },
        std::array<int64_t, 3> strides = { 1, 1, 1 },
        std::array<int64_t, 3> paddings = { 1, 1, 1 },
        std::array<int64_t, 3> dilations = { 1, 1, 1 },
        bool useBias = true,    
        bool useMask = false,
        bool useAutoOffset = true
    );

    void reset_parameters();

    torch::Tensor forward(
        torch::Tensor x,
        std::optional<torch::Tensor> baseOffset = std::nullopt,
        std::optional<torch::Tensor> mask = std::nullopt
    );

private:

    torch::Tensor weight;
    torch::Tensor bias;

    torch::nn::Conv3d convDirs{ nullptr };
    torch::nn::Conv3d convOffsetFromX{ nullptr };
    torch::nn::Conv3d maskConv{ nullptr };

    int64_t in_channels;
    int64_t out_channels;
    std::array<int64_t, 3> kernelSize;
    std::array<int64_t, 3> stride;
    std::array<int64_t, 3> padding;
    std::array<int64_t, 3> dilation;
    int64_t groups;
    int64_t groupsOffset;
    
};

TORCH_MODULE(DeformConv3d);

#endif