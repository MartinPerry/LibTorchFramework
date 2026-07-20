#include "DropPath.h"

//Taken form Timm library

DropPathImpl::DropPathImpl(double dropProb, bool scaleByKeep) : 
    dropProb(dropProb),
    scaleByKeep(scaleByKeep)
{
}

torch::Tensor DropPathImpl::forward(torch::Tensor x)
{
    if (dropProb == 0.0 || !is_training())
    {
        return x;
    }

    const double keepProb = 1.0 - dropProb;

    std::vector<int64_t> shape;
    shape.push_back(x.size(0));

    for (int64_t i = 1; i < x.dim(); ++i)
    {
        shape.push_back(1);
    }

    auto randomTensor = torch::empty(shape, x.options());

    randomTensor.bernoulli_(keepProb);

    if ((keepProb > 0.0) && (scaleByKeep))
    {
        randomTensor.div_(keepProb);
    }

    return x * randomTensor;
}