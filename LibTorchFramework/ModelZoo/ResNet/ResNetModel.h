#ifndef RESNET_MODEL_H
#define RESNET_MODEL_H

#include <memory>
#include <vector>
#include <optional>

#include <torch/torch.h>

#include "../../core/Modules/ModulesOptions.h"


template <typename Activation, typename Normalization, typename ResampleType>
class BasicResidualBlockImpl : public torch::nn::Module
{
public:
    BasicResidualBlockImpl(int64_t in_channels,
        int64_t out_channels,
        int64_t stride = 1,
        int64_t out_expansion = 1
    );

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential conv1{ nullptr };
    torch::nn::Sequential conv2{ nullptr };
    torch::nn::Sequential resample{ nullptr };
    torch::nn::AnyModule actFn;
};

//========================================================================

class ResNetModelImpl : public torch::nn::Module
{
public:
    ResNetModelImpl(std::string type,
        std::vector<int64_t> inputShape,
        std::optional<int64_t> num_classes = std::nullopt,        
        std::vector<int64_t> planes = { 64,128,256,512 },
        std::vector<int64_t> layers = { 3,4,6,3 },
        int64_t out_expansion = 4        
    );

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential preLayer{ nullptr };
    torch::nn::MaxPool2d maxpool{ nullptr };
    torch::nn::ModuleList layersList;
    torch::nn::Sequential fc{ nullptr };
    
    int64_t out_expansion;
    int64_t lastInChannels;

    torch::nn::Sequential _make_layer(        
        int64_t inChannels,
        int64_t blocks,
        int64_t stride);

    torch::Tensor _runLayers(torch::Tensor x);
    void createDefaultFcLayer(std::vector<int64_t> inputShape, std::optional<int64_t> num_classes);
};
TORCH_MODULE(ResNetModel);

//========================================================================
//========================================================================
//========================================================================


template <typename Activation, typename Normalization, typename ResampleType>
BasicResidualBlockImpl<Activation, Normalization, ResampleType>::BasicResidualBlockImpl(int64_t in_channels,
    int64_t out_channels,
    int64_t stride,
    int64_t out_expansion)
{
    int64_t finalOutChannels = out_channels * out_expansion;

    if constexpr (std::is_same<ResampleType, void>::value)    
    {
        conv1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1).bias(false)),
            Normalization(out_channels),
            Activation());        
    }
    else
    {        
        conv1 = torch::nn::Sequential(
            ResampleType(ResampleOptions(out_channels, out_channels, stride).kernelSize(3).padding(1), Normalization(out_channels)),
            Activation());
    }
    register_module("conv1", conv1);

    conv2 = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, finalOutChannels, 3).stride(1).padding(1).bias(false)),
        Normalization(finalOutChannels)
    );

    register_module("conv2", conv2);

    if constexpr (std::is_same<ResampleType, void>::value)
    {
        resample = register_module("resample", torch::nn::Identity());
    }
    else
    {
        resample = register_module("resample", 
            ResampleType(ResampleOptions(in_channels, finalOutChannels, stride), Normalization(finalOutChannels))
        );
    }


    actFn = register_module("actFn", Activation());
}

template <typename Activation, typename Normalization, typename ResampleType>
torch::Tensor BasicResidualBlockImpl<Activation, Normalization, ResampleType>::forward(torch::Tensor x)
{
    auto out = conv1->forward(x);
    out = conv2->forward(out);
    auto residual = resample->forward(x);
    out += residual;
    out = actFn->forward(out);
    return out;
}


#endif