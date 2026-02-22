#include "./ModelInfo.h"

ModelInfo::ModelInfo(const torch::nn::Module& model) : 
    model(model)
{
}


ModelInfo::ModelParams ModelInfo::CountParams() const
{
    int64_t total = 0;
    int64_t trainable = 0;

    for (const auto& p : model.parameters(/*recurse=*/true)) 
    {
        const int64_t n = p.numel();
        total += n;
        if (p.requires_grad()) 
        {
            trainable += n;
        }
    }

    return { trainable, total };
}