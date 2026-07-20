#include "PixelShuffle3D.h"

PixelShuffle3DImpl::PixelShuffle3DImpl(const std::array<int64_t, 3>& scale) : 
    scale(scale)
{
    TORCH_CHECK(scale.size() == 3, "scale must be a 3d tuple");
}

torch::Tensor PixelShuffle3DImpl::forward(torch::Tensor input)
{
    const int64_t batchSize = input.size(0);
    const int64_t channels = input.size(1);
    const int64_t inDepth = input.size(2);
    const int64_t inHeight = input.size(3);
    const int64_t inWidth = input.size(4);

    const int64_t scaleProduct = scale[0] * scale[1] * scale[2];

    const int64_t nOut = channels / scaleProduct;

    const int64_t outDepth = inDepth * scale[0];
    const int64_t outHeight = inHeight * scale[1];
    const int64_t outWidth = inWidth * scale[2];

    auto output = input.contiguous()
        .view({ batchSize, nOut, scale[0], scale[1], scale[2], inDepth, inHeight, inWidth})
        .permute({ 0, 1, 5, 2, 6, 3, 7, 4 })
        .contiguous()
        .view({ batchSize, nOut, outDepth, outHeight, outWidth});

    return output;
}