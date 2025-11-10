#ifndef RESNET_BLOCK_H
#define RESNET_BLOCK_H

#include <torch/torch.h>

#include "./ModulesOptions.h"
#include "./DownSample2d.h"

//======================================================================================
/*
Based on torchvision resnet implementation
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py, but simplified

Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
while original implementation places the stride at the first 1x1 convolution(self.conv1)
according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
This variant is also known as ResNet V1.5 and improves accuracy according to
https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

*/
//======================================================================================



template <typename Activation, typename Normalization, typename ResampleType>
class ResNetBlockImpl : public torch::nn::Module
{
public:
    ResNetBlockImpl(int64_t in_channels,
        int64_t out_channels,
        int64_t stride = 1,        
        int64_t dilation = 1,
        int64_t out_expansion = 1);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential conv1{ nullptr };
    torch::nn::Sequential conv2{ nullptr };
    torch::nn::Sequential conv3{ nullptr };
    torch::nn::AnyModule resample;
    Activation actFn{ nullptr };
};

//=======================================================================================
//=======================================================================================
//=======================================================================================

//custom TORCH_MODULE - 

template <
    typename Activation = torch::nn::ReLU,
    typename Normalization = torch::nn::BatchNorm2d,
    typename ResampleType = void
>
class ResNetBlock :
    public torch::nn::ModuleHolder<ResNetBlockImpl<Activation, Normalization, ResampleType>>
{
public:
    using torch::nn::ModuleHolder<ResNetBlockImpl<Activation, Normalization, ResampleType>>::ModuleHolder;
    using Impl TORCH_UNUSED_EXCEPT_CUDA = ResNetBlockImpl<Activation, Normalization, ResampleType>;
};

//=======================================================================================
//=======================================================================================
//=======================================================================================

template <typename Activation, typename Normalization, typename ResampleType>
ResNetBlockImpl<Activation, Normalization, ResampleType>::ResNetBlockImpl(
    int64_t in_channels,
    int64_t out_channels,
    int64_t stride,    
    int64_t dilation,
    int64_t out_expansion
)
{
    int64_t finalOutChannels = out_channels * out_expansion;
    
    // conv1
    conv1 = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1)
            .stride(1)
            .bias(false)
        ),
        Normalization(out_channels),
        Activation());
    register_module("conv1", conv1);

    // conv2
    if constexpr (std::is_same<ResampleType, void>::value)
    {
        conv2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3)
                .stride(1)
                .padding(dilation)
                .dilation(dilation)
                .bias(false)
            ),
            Normalization(out_channels),
            Activation()
        );
    }
    else
    {
        conv2 = torch::nn::Sequential(
            ResampleType(
                ResampleOptions(out_channels, out_channels, stride).kernelSize(3).padding(dilation).dilation(dilation), 
                Normalization(out_channels)
            ),
            Activation()
        );        
    }
    register_module("conv2", conv2);
    
    // conv3
    conv3 = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, finalOutChannels, 1)
            .stride(1)
            .bias(false)
        ),
        Normalization(finalOutChannels));
    register_module("conv3", conv3);

    // resample
    if constexpr (std::is_same<ResampleType, void>::value)    
    {
        resample = register_module("resample", torch::nn::Identity());        
    }
    else
    {
        resample = register_module("resample", 
            ResampleType(
                ResampleOptions(in_channels, finalOutChannels, stride), 
                Normalization(finalOutChannels)
            )
        );
    }

    actFn = register_module("actFn", Activation());
}

//======================================================================================

template <typename Activation, typename Normalization, typename ResampleType>
torch::Tensor ResNetBlockImpl<Activation, Normalization, ResampleType>::forward(torch::Tensor x)
{
    torch::Tensor out = conv1->forward(x);
    out = conv2->forward(out);
    out = conv3->forward(out);

    torch::Tensor residual = resample.forward(x);

    out = out + residual;
    out = actFn->forward(out);

    return out;
}

#endif

