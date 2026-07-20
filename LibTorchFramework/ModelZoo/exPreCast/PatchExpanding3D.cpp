#include "PatchExpanding3D.h"

PatchExpanding3DImpl::PatchExpanding3DImpl(
    const std::array<int64_t, 3>& patchSize,
    int64_t embedDim,
    int64_t outChans
) : 
    patchSize(patchSize),
    embedDim(embedDim),
    outChans(outChans)
{
    deproj = register_module("deproj", torch::nn::ConvTranspose3d(
            torch::nn::ConvTranspose3dOptions(
                embedDim,
                outChans,
                { patchSize[0], patchSize[1], patchSize[2] })
            .stride({ patchSize[0], patchSize[1], patchSize[2] })));
}

torch::Tensor PatchExpanding3DImpl::forward(torch::Tensor x)
{
    x = deproj->forward(x);

    return x;
}