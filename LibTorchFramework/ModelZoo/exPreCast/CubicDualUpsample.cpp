#include "CubicDualUpsample.h"

CubicDualUpsampleImpl::CubicDualUpsampleImpl(
    int64_t dim,
    const std::array<int64_t, 3>& scale,
    int64_t kernelSize,
    int64_t strideSize,
    int64_t padding
) : 
    dim(dim),
    scale(scale)
{
    scaleFactor = scale[0] * scale[1] * scale[2];

    convP1 = register_module("conv_p1",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(
                dim,
                (scaleFactor / 2) * dim,
                kernelSize)
            .stride(strideSize)
            .padding(padding)
            .bias(false)));

    act = register_module("act", torch::nn::PReLU());

    pixelShuffle = register_module("pixel_shuffle", PixelShuffle3D(scale));

    convP2 = register_module("conv_p2",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(
                dim / 2,
                dim / 2,
                kernelSize)
            .stride(strideSize)
            .padding(padding)
            .bias(false)));

    convB1 = register_module("conv_b1",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(
                dim,
                dim,
                kernelSize)
            .stride(strideSize)
            .padding(padding)));

    /*
    upSample = register_module("up_sample",
        torch::nn::Upsample(
            torch::nn::UpsampleOptions()
            .scale_factor({ static_cast<double>(scale[0]), static_cast<double>(scale[1]), static_cast<double>(scale[2]) })
            .mode(torch::kTrilinear)
            .align_corners(false)));
    */

    convB2 = register_module(
        "conv_b2",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(
                dim,
                dim / 2,
                kernelSize)
            .stride(strideSize)
            .padding(padding)
            .bias(false)));

    convMerge = register_module(
        "conv_merge",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(
                dim,
                dim / 2,
                kernelSize)
            .stride(strideSize)
            .padding(padding)
            .bias(false)));

    norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dim / 2 })));
}

torch::Tensor CubicDualUpsampleImpl::forward(
    torch::Tensor x)
{
    // Input:
    // Python: (B,T,H,W,C)
    // C++ internal Conv3d format: (B,C,T,H,W)

    using namespace torch::nn::functional;

    auto scaleFactorDims = std::vector<double>{ 
            static_cast<double>(scale[0]), 
            static_cast<double>(scale[1]), 
            static_cast<double>(scale[2]) 
    };

    InterpolateFuncOptions opts = InterpolateFuncOptions().
        scale_factor(std::move(scaleFactorDims)).
        mode(torch::kTrilinear).
        align_corners(false);

    x = x.permute({ 0, 4, 1, 2, 3}).contiguous();

    auto xP = convP1->forward(x);
    xP = act->forward(xP);
    xP = pixelShuffle->forward(xP);
    xP = convP2->forward(xP);

    auto xB = convB1->forward(x);
    xB = act->forward(xB);
    xB = interpolate(xB, opts);
    //xB = upSample->forward(xB);
    xB = convB2->forward(xB);

    x = torch::cat({ xP, xB }, 1);

    x = convMerge->forward(x);

    x = x.permute({ 0, 2, 3, 4, 1}).contiguous();

    if (norm.is_empty() == false)
    {
        x = norm->forward(x);
    }

    return x;
}