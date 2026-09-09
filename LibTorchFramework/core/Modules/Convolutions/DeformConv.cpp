#include "./DeformConv.h"

#include "./DeformConvImpl/torchvision/vision_deform_conv2d.h"

#include "./DeformConvImpl/tvdcn/ops/deform_conv3d.h"

//==================================================================================================
// DeformConv3d
//==================================================================================================


DeformConv2dImpl::DeformConv2dImpl(
    int64_t in_channels,
    int64_t out_channels,
    std::array<int64_t, 2> kernelSizes,
    std::array<int64_t, 2> strides,
    std::array<int64_t, 2> paddings,
    std::array<int64_t, 2> dilations,
    bool useBias,
    bool useMask,
    bool useAutoOffset) : 
    in_channels(in_channels),
    out_channels(out_channels),
    kernelSize(kernelSizes),
    stride(strides),
    padding(paddings),
    dilation(dilations),
    groups(1),
    groupsOffset(1),
    useMask(useMask)
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
        torch::empty({ out_channels, in_channels / groups, kernelSize[0], kernelSize[1] })
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
            torch::nn::Conv2dOptions(in_channels, 2 * kernelSize[0] * kernelSize[1], kernelSize)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .padding_mode(torch::kReplicate)
            .bias(true)
        );
        register_module("convOffsetFromX", convOffsetFromX);
    }
    else
    {
        convDirs = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(2, 2 * kernelSize[0] * kernelSize[1], kernelSize)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .padding_mode(torch::kReplicate)
            .bias(true)
        );
        register_module("convDirs", convDirs);
    }

    if (useMask)
    {
        maskConv = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, kernelSize[0] * kernelSize[1], kernelSize)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
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
        auto fan_in = in_channels * kernelSize[0] * kernelSize[1];
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

    if (mask.has_value() == false)
    {
        if (maskConv.is_empty() == false)
        {
            mask = maskConv->forward(x);
            mask = torch::sigmoid(*mask);
        }
        else
        {
            // Fallback
            mask = torch::zeros_like(x);
        }
    }

    
    torch::Tensor out = vision::ops::deform_conv2d(
        x, weight, offset, *mask, bias,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups, groupsOffset,
        useMask
    );


    return out;
}

torch::Tensor DeformConv2dImpl::forward(torch::Tensor x)
{
    return this->forward(x, std::nullopt, std::nullopt);
}

//==================================================================================================
// DeformConv3d
//==================================================================================================


DeformConv3dImpl::DeformConv3dImpl(
    int64_t in_channels,
    int64_t out_channels,
    std::array<int64_t, 3> kernelSizes, //w, h, d
    std::array<int64_t, 3> strides,   //w, h, d
    std::array<int64_t, 3> paddings,  //w, h, d
    std::array<int64_t, 3> dilations, //w, h, d     
    bool useMask,
    bool useAutoOffset) :
    in_channels(in_channels),
    out_channels(out_channels),
    kernelSize(kernelSizes),
    stride(strides),
    padding(paddings),
    dilation(dilations),
    groups(1),
    groupsOffset(1)    
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
        torch::empty({ out_channels, in_channels / groups, 
            kernelSize[0], kernelSize[1], kernelSize[2]
        })
    );

    bias = register_parameter("bias", torch::empty(out_channels));
    

    this->reset_parameters();

    int64_t ks = kernelSize[0] * kernelSize[1] * kernelSize[2];

    if (useAutoOffset)
    {
        convOffsetFromX = torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, 3 * ks, kernelSize)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .padding_mode(torch::kReplicate)
            .bias(true)
        );
        convOffsetFromX = register_module("convOffsetFromX", convOffsetFromX);
    }
    else
    {
        convDirs = torch::nn::Conv3d(
            torch::nn::Conv3dOptions(2, 3 * ks, kernelSize)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .padding_mode(torch::kReplicate)
            .bias(true)
        );
        convDirs = register_module("convDirs", convDirs);
    }

    if (useMask)
    {        
        maskConv = torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, ks, kernelSize)
            .stride(stride)
            .padding(padding)
            .dilation(dilation)
            .bias(true)
        );
        maskConv = register_module("maskConv", maskConv);

        // Xavier init equivalent
        torch::nn::init::xavier_uniform_(maskConv->weight);
        torch::nn::init::zeros_(maskConv->bias);        
    }


}

void DeformConv3dImpl::reset_parameters()
{
    torch::nn::init::kaiming_uniform_(weight, std::sqrt(5));

    auto fan_in = in_channels * kernelSize[0] * kernelSize[1] * kernelSize[2];
    auto bound = 1.0 / std::sqrt(fan_in);
    torch::nn::init::uniform_(bias, -bound, bound);
    
}

std::array<int64_t, 3> DeformConv3dImpl::CalcOutputSize(torch::Tensor x) const
{
    //x => [D, H, W]

    int d = (x.size(2) + 2 * padding[2] - dilation[2] * (kernelSize[2] - 1) - 1) / stride[2] + 1;
    int h = (x.size(3) + 2 * padding[1] - dilation[1] * (kernelSize[1] - 1) - 1) / stride[1] + 1;
    int w = (x.size(4) + 2 * padding[0] - dilation[0] * (kernelSize[0] - 1) - 1) / stride[0] + 1;

    return { w, h, d };
}

torch::Tensor DeformConv3dImpl::forward(
    torch::Tensor x,
    std::optional<torch::Tensor> baseOffset,
    std::optional<torch::Tensor> mask)
{
    //x.size(1) => channels

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
        int N = x.size(0);
        int64_t ks = kernelSize[0] * kernelSize[1] * kernelSize[2];
        auto [w, h, d] = this->CalcOutputSize(x);
        offset = torch::zeros({ N, 3 * ks, d, h, w });
    }

    if (mask.has_value() == false)
    {
        if (maskConv.is_empty() == false)
        {
            mask = maskConv->forward(x);
            mask = torch::sigmoid(*mask);
        }     
        else
        {
            // Fallback
            int N = x.size(0);
            int64_t ks = kernelSize[0] * kernelSize[1] * kernelSize[2];
            auto [w, h, d] = this->CalcOutputSize(x);
            mask = torch::zeros({ N, ks, d, h, w }, x.options());
        }
    }
    
    //it crashes in backward if offset, mask and/or bias is nullopt

    torch::Tensor out = tvdcn::ops::deform_conv3d(
        x, weight, offset, mask, bias,
        stride,
        padding,
        dilation,
        groups
    );


    return out;
}

torch::Tensor DeformConv3dImpl::forward(torch::Tensor x)
{
    return this->forward(x, std::nullopt, std::nullopt);
}