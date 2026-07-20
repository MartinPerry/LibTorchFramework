#include "WindowUtils.h"

torch::Tensor windowPartition(
    torch::Tensor x,
    const std::array<int64_t, 3>& windowSize)
{
    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t C = x.size(4);

    x = x.view({ B, D / windowSize[0], windowSize[0], H / windowSize[1], windowSize[1], W / windowSize[2], windowSize[2], C });

    x = x.permute( { 0, 1, 3, 5, 2, 4, 6, 7});

    x = x.contiguous();

    const int64_t windowVolume = windowSize[0] * windowSize[1] * windowSize[2];

    return x.view({-1, windowVolume, C});
}

torch::Tensor windowReverse(
    torch::Tensor windows,
    const std::array<int64_t, 3>& windowSize,
    int64_t B,
    int64_t D,
    int64_t H,
    int64_t W)
{
    windows = windows.view( {B, D / windowSize[0], H / windowSize[1], W / windowSize[2], 
        windowSize[0], windowSize[1], windowSize[2], -1});

    windows = windows.permute({0, 1, 4, 2, 5, 3, 6, 7 });

    windows = windows.contiguous();

    return windows.view({ B, D, H, W, -1});
}

std::array<int64_t, 3> getWindowSize(
    const std::array<int64_t, 3>& xSize,
    const std::array<int64_t, 3>& windowSize)
{
    std::array<int64_t, 3> result = windowSize;

    for (size_t i = 0; i < 3; ++i)
    {
        if (xSize[i] <= windowSize[i])
        {
            result[i] = xSize[i];
        }
    }

    return result;
}

std::pair<
    std::array<int64_t, 3>,
    std::array<int64_t, 3>>
    getWindowSize(
        const std::array<int64_t, 3>& xSize,
        const std::array<int64_t, 3>& windowSize,
        const std::array<int64_t, 3>& shiftSize)
{
    std::array<int64_t, 3> useWindow = windowSize;
    std::array<int64_t, 3> useShift = shiftSize;

    for (size_t i = 0; i < 3; ++i)
    {
        if (xSize[i] <= windowSize[i])
        {
            useWindow[i] = xSize[i];
            useShift[i] = 0;
        }
    }

    return { useWindow, useShift };
}