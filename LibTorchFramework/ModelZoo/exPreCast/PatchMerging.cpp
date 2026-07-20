#include "PatchMerging.h"

using namespace torch::indexing;

PatchMergingImpl::PatchMergingImpl(int64_t dim) : 
    dim(dim)
{
    reduction = register_module("reduction",
        torch::nn::Linear(torch::nn::LinearOptions(4 * dim, 2 * dim).bias(false)));

    norm = register_module("norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({ 4 * dim })));
}

torch::Tensor PatchMergingImpl::forward(torch::Tensor x)
{
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    const bool padInput = (H % 2 == 1) || (W % 2 == 1);

    if (padInput)
    {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({ 0, 0, 0, W % 2, 0, H % 2 }));
    }

    auto x0 = x.index({ Slice(), Slice(), Slice(0, None, 2), Slice(0, None, 2), Slice() });
    auto x1 = x.index({ Slice(), Slice(), Slice(1, None, 2), Slice(0, None, 2), Slice() });
    auto x2 = x.index({ Slice(), Slice(), Slice(0, None, 2), Slice(1, None, 2), Slice() });
    auto x3 = x.index({ Slice(), Slice(), Slice(1, None, 2), Slice(1, None, 2), Slice()});

    x = torch::cat({ x0, x1, x2, x3 }, -1);

    x = norm->forward(x);
    x = reduction->forward(x);

    return x;
}