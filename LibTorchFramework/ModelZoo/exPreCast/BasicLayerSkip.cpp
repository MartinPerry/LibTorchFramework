#include "BasicLayerSkip.h"

#include <cmath>

using namespace torch::indexing;


BasicLayerSkipImpl::BasicLayerSkipImpl(
    int64_t dim,
    int64_t depth,
    int64_t numHeads,
    const std::array<int64_t, 3>& windowSize,
    double mlpRatio,
    bool qkvBias,
    std::optional<double> qkScale,
    double drop,
    double attnDrop,
    double dropPath,
    bool useSubsample
) : 
    windowSize(windowSize),
    depth(depth),
    useCheckpoint(useCheckpoint),
    blocks(register_module("blocks", torch::nn::ModuleList()))
{
    shiftSize = 
    {
        windowSize[0] / 2,
        windowSize[1] / 2,
        windowSize[2] / 2
    };

    for (int64_t i = 0; i < depth; ++i)
    {
        auto block = SwinTransformerBlock3D(
            dim,
            numHeads,
            windowSize,
            (i % 2 == 0) ?
            std::array<int64_t, 3>{ 0, 0, 0 } :
            shiftSize,
            mlpRatio,
            qkvBias,
            qkScale,
            drop,
            attnDrop,
            dropPath
        );

        blocks->push_back(block);
    }

    if (useSubsample)
    {
        subsample = register_module("subsample", PatchMerging(dim));
    }
}


torch::Tensor BasicLayerSkipImpl::computeMask(
    int64_t D,
    int64_t H,
    int64_t W,
    const std::array<int64_t, 3>& windowSize,
    const std::array<int64_t, 3>& shiftSize,
    torch::Device device)
{
    auto imgMask = torch::zeros({ 1, D, H, W, 1 }, torch::TensorOptions().device(device));

    int cnt = 0;

    std::vector<std::pair<int64_t, int64_t>> dRanges =
    {
        { D - windowSize[0], D },
        { D - windowSize[0], D - shiftSize[0] },
        { D - shiftSize[0], D }
    };

    std::vector<std::pair<int64_t, int64_t>> hRanges =
    {
        { H - windowSize[1], H },
        { H - windowSize[1], H - shiftSize[1] },
        { H - shiftSize[1], H }
    };

    std::vector<std::pair<int64_t, int64_t>> wRanges =
    {
        { W - windowSize[2], W },
        { W - windowSize[2], W - shiftSize[2] },
        { W - shiftSize[2], W }
    };

    for (auto d : dRanges)
    {
        for (auto h : hRanges)
        {
            for (auto w : wRanges)
            {
                imgMask.index_put_(
                    {
                        Slice(),
                        Slice(d.first, d.second),
                        Slice(h.first, h.second),
                        Slice(w.first, w.second),
                        Slice()
                    },
                    cnt);

                cnt++;
            }
        }
    }

    auto maskWindows = windowPartition(imgMask,windowSize).squeeze(-1);

    auto attnMask = maskWindows.unsqueeze(1) - maskWindows.unsqueeze(2);

    attnMask = attnMask.masked_fill(attnMask != 0, -100.0);

    attnMask = attnMask.masked_fill(attnMask == 0, 0.0);

    return attnMask;
}


std::tuple<torch::Tensor, torch::Tensor>
BasicLayerSkipImpl::forward(
    torch::Tensor x)
{
    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t D = x.size(2);
    const int64_t H = x.size(3);
    const int64_t W = x.size(4);

    auto result = getWindowSize( { D, H, W }, windowSize, shiftSize);

    auto win = result.first;
    auto shift = result.second;

    x = x.permute({ 0, 2, 3, 4, 1 }).contiguous();

    const int64_t Dp = ((D + win[0] - 1) / win[0]) * win[0];
    const int64_t Hp = ((H + win[1] - 1) / win[1]) * win[1];
    const int64_t Wp = ((W + win[2] - 1) / win[2]) * win[2];

    auto attnMask = computeMask( Dp, Hp, Wp, win, shift, x.device());

    for (auto& module : *blocks)
    {
        auto blk = module->as<SwinTransformerBlock3DImpl>();

        x = blk->forward(x, attnMask);
    }

    x = x.view({ B, D, H, W, -1 });

    auto xSkip = x.permute({ 0, 4, 1, 2, 3}).contiguous();

    if (subsample.is_empty() == false)
    {
        x = subsample->forward(x);
    }

    x = x.permute({ 0, 4, 1, 2, 3 }).contiguous();

    return { x, xSkip };
}