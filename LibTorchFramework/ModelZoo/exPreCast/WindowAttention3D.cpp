#include "WindowAttention3D.h"

#include <cmath>

namespace
{

    void truncNormal(
        torch::Tensor tensor,
        double std = 0.02)
    {
        // TODO: Replace with a true truncated normal implementation if available.
        torch::NoGradGuard guard;
        tensor.normal_(0.0, std);
    }

}

WindowAttention3DImpl::WindowAttention3DImpl(
    int64_t dim,
    const std::array<int64_t, 3>& windowSize,
    int64_t numHeads,
    bool qkvBias,
    std::optional<double> qkScale,
    double attnDropProb,
    double projDropProb
) : 
    dim(dim),
    windowSize(windowSize),
    numHeads(numHeads)
{
    const int64_t headDim = dim / numHeads;

    if (qkScale.has_value())
    {
        scale = *qkScale;
    }
    else
    {
        scale = 1.0 / std::sqrt(static_cast<double>(headDim));
    }

    const int64_t tableSize =
        (2 * windowSize[0] - 1) *
        (2 * windowSize[1] - 1) *
        (2 * windowSize[2] - 1);

    relativePositionBiasTable = register_parameter(
        "relative_position_bias_table",
        torch::zeros({ tableSize, numHeads }));

    auto coordsD = torch::arange(windowSize[0], torch::kLong);
    auto coordsH = torch::arange(windowSize[1], torch::kLong);
    auto coordsW = torch::arange(windowSize[2], torch::kLong);

    auto mesh = torch::meshgrid(
        { coordsD, coordsH, coordsW },
        "ij");

    auto coords = torch::stack(mesh);
    auto coordsFlatten = coords.flatten(1);

    auto relativeCoords =
        coordsFlatten.unsqueeze(2) -
        coordsFlatten.unsqueeze(1);

    relativeCoords =
        relativeCoords.permute({ 1, 2, 0 }).contiguous();

    relativeCoords.index_put_(
        { "...", 0 },
        relativeCoords.index({ "...", 0 }) + windowSize[0] - 1);

    relativeCoords.index_put_(
        { "...", 1 },
        relativeCoords.index({ "...", 1 }) + windowSize[1] - 1);

    relativeCoords.index_put_(
        { "...", 2 },
        relativeCoords.index({ "...", 2 }) + windowSize[2] - 1);

    relativeCoords.index_put_(
        { "...", 0 },
        relativeCoords.index({ "...", 0 }) *
        ((2 * windowSize[1] - 1) * (2 * windowSize[2] - 1)));

    relativeCoords.index_put_(
        { "...", 1 },
        relativeCoords.index({ "...", 1 }) *
        (2 * windowSize[2] - 1));

    relativePositionIndex = relativeCoords.sum(-1);

    register_buffer(
        "relative_position_index",
        relativePositionIndex);

    qkv = register_module(
        "qkv",
        torch::nn::Linear(
            torch::nn::LinearOptions(dim, dim * 3).bias(qkvBias)));

    attnDrop = register_module(
        "attn_drop",
        torch::nn::Dropout(attnDropProb));

    proj = register_module(
        "proj",
        torch::nn::Linear(dim, dim));

    projDrop = register_module(
        "proj_drop",
        torch::nn::Dropout(projDropProb));

    softmax = register_module(
        "softmax",
        torch::nn::Softmax(-1));

    truncNormal(relativePositionBiasTable, 0.02);
}

torch::Tensor WindowAttention3DImpl::forward(
    torch::Tensor x,
    std::optional<torch::Tensor> mask)
{
    const int64_t B = x.size(0);
    const int64_t N = x.size(1);
    const int64_t C = x.size(2);

    auto qkvTensor =
        qkv->forward(x)
        .reshape({ B, N, 3, numHeads, C / numHeads })
        .permute({ 2, 0, 3, 1, 4 });

    auto q = qkvTensor[0];
    auto k = qkvTensor[1];
    auto v = qkvTensor[2];

    q = q * scale;

    auto attn = torch::matmul(q, k.transpose(-2, -1));

    auto relativeBias =
        relativePositionBiasTable
        .index_select(
            0,
            relativePositionIndex
            .index({ torch::indexing::Slice(0, N),
                     torch::indexing::Slice(0, N) })
            .reshape(-1))
        .reshape({ N, N, numHeads });

    relativeBias = relativeBias.permute({ 2, 0, 1 }).contiguous();

    attn = attn + relativeBias.unsqueeze(0);

    if (mask.has_value())
    {
        auto m = *mask;
        const int64_t nW = m.size(0);

        attn = attn.view({ B / nW, nW, numHeads, N, N }) + m.unsqueeze(1).unsqueeze(0);

        attn = attn.view({ -1, numHeads, N, N });

        attn = softmax->forward(attn);
    }
    else
    {
        attn = softmax->forward(attn);
    }

    attn = attnDrop->forward(attn);

    x = torch::matmul(attn, v) .transpose(1, 2).reshape({ B, N, C });

    x = proj->forward(x);
    x = projDrop->forward(x);

    return x;
}