#ifndef DEFORM_CONV_H
#define DEFORM_CONV_H

#include <utility>
#include <optional>

#include <torch/torch.h>


class DeformConv2dImpl : public torch::nn::Module
{
public:

    DeformConv2dImpl(
        int64_t in_channels,
        int64_t out_channels,
        std::pair<int64_t, int64_t> kernelSize = { 3, 3 },
        std::pair<int64_t, int64_t> stride = { 1, 1 },
        std::pair<int64_t, int64_t> padding = { 1, 1 },
        std::pair<int64_t, int64_t> dilation = { 1, 1 },
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
    std::pair<int64_t, int64_t> kernelSize;
    std::pair<int64_t, int64_t> stride;
    std::pair<int64_t, int64_t> padding;
    std::pair<int64_t, int64_t> dilation;
    int64_t groups;
    int64_t groupsOffset;

    bool useMask;
    bool useAutoOffset;
};

TORCH_MODULE(DeformConv2d);


#endif