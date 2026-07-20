#include "PatchEmbed3D.h"

using namespace torch::indexing;

PatchEmbed3DImpl::PatchEmbed3DImpl(
    const std::array<int64_t, 3>& patchSize,
    int64_t inChans,
    int64_t embedDim,
    bool useNorm
) : 
    patchSize(patchSize),
    inChans(inChans),
    embedDim(embedDim)
{
    proj = register_module("proj", torch::nn::Conv3d(
            torch::nn::Conv3dOptions(
                inChans,
                embedDim,
                { patchSize[0], patchSize[1], patchSize[2] })
            .stride({ patchSize[0], patchSize[1], patchSize[2] })));

    if (useNorm)
    {
        norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ embedDim })));
    }
}

torch::Tensor PatchEmbed3DImpl::forward(torch::Tensor x)
{
    const int64_t D = x.size(2);
    const int64_t H = x.size(3);
    const int64_t W = x.size(4);

    if (W % patchSize[2] != 0)
    {
        x = torch::nn::functional::pad(
            x, 
            torch::nn::functional::PadFuncOptions({ 0, patchSize[2] - W % patchSize[2] })
        );
    }

    if (H % patchSize[1] != 0)
    {
        x = torch::nn::functional::pad(
            x,
            torch::nn::functional::PadFuncOptions({ 0, 0, 0, patchSize[1] - H % patchSize[1]})
        );
    }

    if (D % patchSize[0] != 0)
    {
        x = torch::nn::functional::pad(
            x,
            torch::nn::functional::PadFuncOptions({ 0, 0, 0, 0, 0, patchSize[0] - D % patchSize[0] })
        );
    }

    x = proj->forward(x);

    if (norm.is_empty() == false)
    {
        const int64_t outD = x.size(2);
        const int64_t outH = x.size(3);
        const int64_t outW = x.size(4);

        x = x.flatten(2).transpose(1, 2);

        x = norm->forward(x);

        x = x.transpose(1, 2).view({-1, embedDim, outD, outH, outW });
    }

    return x;
}