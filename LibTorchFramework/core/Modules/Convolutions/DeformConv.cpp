#include "./DeformConv.h"

#include "./DeformConvImpl/deform_conv2d.h"

DeformConv2dImpl::DeformConv2dImpl(
    int64_t in_channels,
    int64_t out_channels,
    std::pair<int64_t, int64_t> kernelSize,
    std::pair<int64_t, int64_t> stride,
    std::pair<int64_t, int64_t> padding,
    std::pair<int64_t, int64_t> dilation,
    bool useBias,
    bool useMask,
    bool useAutoOffset) : 
    in_channels(in_channels),
    out_channels(out_channels),
    kernelSize(kernelSize),
    stride(stride),
    padding(padding),
    dilation(dilation),
    groups(1),
    groupsOffset(1),
    useMask(useMask), 
    useAutoOffset(useAutoOffset)
{
    if (in_channels % groups != 0) 
    { 
        throw std::runtime_error("in_channels must be divisible by groups"); 
    }
    if (out_channels % groups != 0) 
    { 
        throw std::runtime_error("out_channels must be divisible by groups"); 
    }


    weight = register_parameter(
        "weight",
        torch::empty({ out_channels, in_channels / groups, kernelSize.first, kernelSize.second })
    );

    if (useBias)
    {
        bias = register_parameter("bias", torch::empty(out_channels));
    }
    else
    {
        bias = torch::Tensor();
    }

    this->reset_parameters();

    if (useAutoOffset)
    {       
        convOffsetFromX = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, 2 * kernelSize.first * kernelSize.second, { kernelSize.first, kernelSize.second })
            .stride({ stride.first, stride.second })
            .padding({ padding.first, padding.second })
            .dilation({ dilation.first, dilation.second })
            .bias(true)
        );
        register_module("convOffsetFromX", convOffsetFromX);
    }
    else
    {
        convDirs = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(2, 2 * kernelSize.first * kernelSize.second, { kernelSize.first, kernelSize.second })
            .stride({ stride.first, stride.second })
            .padding({ padding.first, padding.second })
            .dilation({ dilation.first, dilation.second })
            .bias(true)
        );
        register_module("convDirs", convDirs);
    }

    if (useMask)
    {
        maskConv = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, kernelSize.first * kernelSize.second, { kernelSize.first, kernelSize.second })
            .stride({ stride.first, stride.second })
            .padding({ padding.first, padding.second })
            .dilation({ dilation.first, dilation.second })
            .bias(true)
        );
        // Xavier init equivalent
        torch::nn::init::xavier_uniform_(maskConv->weight);
        torch::nn::init::zeros_(maskConv->bias);
        register_module("maskConv", maskConv);
    }
 
    
}

void DeformConv2dImpl::reset_parameters()
{
    torch::nn::init::kaiming_uniform_(weight, std::sqrt(5));

    if (bias.defined())
    {
        auto fan_in = in_channels * kernelSize.first * kernelSize.second;
        auto bound = 1.0 / std::sqrt(fan_in);
        torch::nn::init::uniform_(bias, -bound, bound);
    }
}

torch::Tensor DeformConv2dImpl::forward(
    torch::Tensor x,
    std::optional<torch::Tensor> baseOffset,
    std::optional<torch::Tensor> mask)
{
    torch::Tensor offset;
    
    if ((convDirs.is_empty() == false) && (baseOffset.has_value()))
    {
        offset = convDirs->forward(*baseOffset);
    }
    else if (convOffsetFromX.is_empty() == false)
    {
        offset = convOffsetFromX->forward(x);
    }
    else
    {
        // Fallback
        offset = torch::zeros_like(x);
    }

    if ((maskConv.is_empty() == false) && (mask.has_value() == false))
    {
        mask = maskConv->forward(x);
        mask = torch::sigmoid(*mask);
    }

    // DeformConv placeholder
    // TODO: Replace with actual deformable convolution
    //torch::Tensor out = x; // Identity until implemented

    torch::Tensor out = vision::ops::deform_conv2d(
        x, weight, offset, *mask, bias,
        stride.first, stride.second,
        padding.first, padding.second,
        dilation.first, dilation.second,
        groups, groupsOffset,
        useMask
    );


    return out;
}