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

ModelInfo::MemoryInfo ModelInfo::GetMemorySize() const
{
    MemoryInfo total = { 0 };

    for (const auto& p : model.parameters())
    {
        auto bytes = p.numel() * p.element_size();
        if (p.is_cpu())
        {
            total.cpuBytes += bytes;
        }
        else if (p.is_cuda())
        {
            total.gpuBytes += bytes;
        }

        if (p.grad().defined())
        {
            auto g = p.grad();
            auto gbytes = g.numel() * g.element_size();
            if (g.is_cpu())
            {
                total.cpuBytes += gbytes;
            }
            else if (g.is_cuda())
            {
                total.gpuBytes += gbytes;
            }
        }
    }

    for (const auto& b : model.buffers())
    {
        auto bytes = b.numel() * b.element_size();
        if (b.is_cpu())
        {
            total.cpuBytes += bytes;
        }
        else if (b.is_cuda())
        {
            total.gpuBytes += bytes;
        }
    }

    return total;
}