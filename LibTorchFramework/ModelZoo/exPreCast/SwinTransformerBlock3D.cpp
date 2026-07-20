#include "SwinTransformerBlock3D.h"

#include "./WindowUtils.h"

using namespace torch::indexing;

SwinTransformerBlock3DImpl::SwinTransformerBlock3DImpl(
    int64_t dim,
    int64_t numHeads,
    const std::array<int64_t, 3>& windowSize,
    const std::array<int64_t, 3>& shiftSize,
    double mlpRatio,
    bool qkvBias,
    std::optional<double> qkScale,
    double drop,
    double attnDrop,
    double dropPath
) : 
    dim(dim),
    numHeads(numHeads),
    windowSize(windowSize),
    shiftSize(shiftSize),
    mlpRatio(mlpRatio)
{
    TORCH_CHECK(
        shiftSize[0] >= 0 && shiftSize[0] < windowSize[0],
        "shift_size must be in [0, window_size)");

    TORCH_CHECK(
        shiftSize[1] >= 0 && shiftSize[1] < windowSize[1],
        "shift_size must be in [0, window_size)");

    TORCH_CHECK(
        shiftSize[2] >= 0 && shiftSize[2] < windowSize[2],
        "shift_size must be in [0, window_size)");

    norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dim })));

    attn = register_module("attn", WindowAttention3D(
            dim,
            windowSize,
            numHeads,
            qkvBias,
            qkScale,
            attnDrop,
            drop)
    );

    if (dropPath > 0.0)
    {
        dropPathLayer = register_module("drop_path", DropPath(dropPath));     // TODO
    }
    else
    {
        identity = register_module("drop_path", torch::nn::Identity());
    }

    norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dim })));

    const int64_t hiddenDim = static_cast<int64_t>(dim * mlpRatio);

    mlp = register_module("mlp", Mlp(dim, hiddenDim, std::nullopt, drop));
}

torch::Tensor SwinTransformerBlock3DImpl::forwardPart1(
    torch::Tensor x,
    torch::Tensor maskMatrix)
{
    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t C = x.size(4);

    auto sizes = getWindowSize({ D, H, W }, windowSize, shiftSize);        // TODO

    auto win = sizes.first;
    auto shift = sizes.second;

    x = norm1->forward(x);

    const int64_t padD1 = (win[0] - D % win[0]) % win[0];
    const int64_t padB = (win[1] - H % win[1]) % win[1];
    const int64_t padR = (win[2] - W % win[2]) % win[2];

    x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({ 0, 0, 0, padR, 0, padB, 0, padD1 }));

    const int64_t Dp = x.size(1);
    const int64_t Hp = x.size(2);
    const int64_t Wp = x.size(3);

    torch::Tensor shiftedX;
    std::optional<torch::Tensor> attnMask;

    if ((shift[0] > 0) || (shift[1] > 0) || (shift[2] > 0))
    {
        shiftedX =  torch::roll(
                x,
                { -shift[0], -shift[1], -shift[2] },
                { 1, 2, 3 });

        attnMask = maskMatrix;
    }
    else
    {
        shiftedX = x;
    }

    auto windows = windowPartition(shiftedX, win);      // TODO

    auto attnWindows = attn->forward(windows, attnMask);

    attnWindows = attnWindows.view({ -1, win[0], win[1], win[2], C });

    shiftedX = windowReverse(attnWindows, win, B, Dp, Hp, Wp);      // TODO

    if ((shift[0] > 0) || (shift[1] > 0) || (shift[2] > 0))
    {
        x = torch::roll(shiftedX, { shift[0], shift[1], shift[2] }, { 1, 2, 3 });
    }
    else
    {
        x = shiftedX;
    }

    if ((padD1 > 0) || (padB > 0) || (padR > 0))
    {
        x = x.index(
                {
                    Slice(),
                    Slice(0, D),
                    Slice(0, H),
                    Slice(0, W),
                    Slice()
                }).contiguous();
    }

    return x;
}

torch::Tensor SwinTransformerBlock3DImpl::forwardPart2(
    torch::Tensor x)
{
    x = norm2->forward(x);
    x = mlp->forward(x);

    if (dropPathLayer.is_empty() == false)
    {
        return dropPathLayer->forward(x);
    }

    return identity->forward(x);
}

torch::Tensor SwinTransformerBlock3DImpl::forward(
    torch::Tensor x,
    torch::Tensor maskMatrix)
{
    auto shortcut = x;
    
    x = forwardPart1(x, maskMatrix);

    if (dropPathLayer.is_empty() == false)
    {
        x = shortcut + dropPathLayer->forward(x);
    }
    else
    {
        x = shortcut + identity->forward(x);
    }

    x = x + forwardPart2(x);

    return x;
}