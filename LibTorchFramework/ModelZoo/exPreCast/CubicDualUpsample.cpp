#include "CubicDualUpsample.h"


CubicDualUpsampleImpl::CubicDualUpsampleImpl(
    int64_t dim,
    const std::array<int64_t, 3>& scale,
    int64_t kernelSize,
    int64_t strideSize,
    int64_t padding
) :    
    scale(scale)
{
    int64_t scaleFactor = scale[0] * scale[1] * scale[2];

    act = register_module("act", torch::nn::PReLU());

    pixelShuffle = register_module("pixel_shuffle", PixelShuffle3D(scale));

    convP1 = register_module("conv_p1",
        CreateDefaultConv3d(dim, (scaleFactor / 2) * dim, kernelSize, strideSize, padding, false));
            
    convP2 = register_module("conv_p2",
        CreateDefaultConv3d(dim / 2, dim / 2, kernelSize, strideSize, padding, false));

    convB1 = register_module("conv_b1",
        CreateDefaultConv3d(dim, dim, kernelSize, strideSize, padding, true));
    
    convB2 = register_module("conv_b2",
        CreateDefaultConv3d(dim, dim / 2, kernelSize, strideSize, padding, false));
        
    convMerge = register_module("conv_merge", 
        CreateDefaultConv3d(dim, dim / 2, kernelSize, strideSize, padding, false));

    norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dim / 2 })));
}

torch::nn::Conv3d CubicDualUpsampleImpl::CreateDefaultConv3d(int64_t inC, int64_t outC,
    int64_t kernelSize,
    int64_t strideSize,
    int64_t padding,
    bool bias)
{
    return torch::nn::Conv3d(
        torch::nn::Conv3dOptions(inC, outC, kernelSize)
        .stride(strideSize)
        .padding(padding)
        .bias(bias)
    );
}


StackDeformConv3d CubicDualUpsampleImpl::CreateDeformConv3d(int64_t inC, int64_t outC,
    int64_t kernelSize,
    int64_t strideSize,
    int64_t padding,
    bool bias)
{
    return StackDeformConv3d(inC, outC,
        std::array<int64_t, 3>{ kernelSize , kernelSize , kernelSize },
        std::array<int64_t, 3>{ strideSize , strideSize , strideSize },
        std::array<int64_t, 3>{ padding, padding, padding },
        std::array<int64_t, 3>{ 1, 1, 1 },
        bias
    );
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

    auto xP = convP1.forward(x);
    xP = act->forward(xP);
    xP = pixelShuffle->forward(xP);
    xP = convP2.forward(xP);

    auto xB = convB1.forward(x);
    xB = act->forward(xB);
    xB = torch::nn::functional::interpolate(xB, opts);    
    xB = convB2.forward(xB);

    x = torch::cat({ xP, xB }, 1);

    x = convMerge->forward(x);

    x = x.permute({ 0, 2, 3, 4, 1}).contiguous();

    if (norm.is_empty() == false)
    {
        x = norm->forward(x);
    }

    return x;
}