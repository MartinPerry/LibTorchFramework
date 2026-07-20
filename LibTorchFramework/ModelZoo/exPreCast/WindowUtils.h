#pragma once

#include <torch/torch.h>
#include <array>
#include <utility>

torch::Tensor windowPartition(
    torch::Tensor x,
    const std::array<int64_t, 3>& windowSize);

torch::Tensor windowReverse(
    torch::Tensor windows,
    const std::array<int64_t, 3>& windowSize,
    int64_t B,
    int64_t D,
    int64_t H,
    int64_t W);

std::array<int64_t, 3> getWindowSize(
    const std::array<int64_t, 3>& xSize,
    const std::array<int64_t, 3>& windowSize);

std::pair<
    std::array<int64_t, 3>,
    std::array<int64_t, 3>>
    getWindowSize(
        const std::array<int64_t, 3>& xSize,
        const std::array<int64_t, 3>& windowSize,
        const std::array<int64_t, 3>& shiftSize);