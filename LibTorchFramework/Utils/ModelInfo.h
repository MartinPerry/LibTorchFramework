#ifndef MODEL_INFO_H
#define MODEL_INFO_H

#include <cstdint>

#include <torch/torch.h>

class ModelInfo
{
public:
    struct ModelParams
    {
        int64_t trainable;
        int64_t total;
    };

    struct MemoryInfo
    {
        size_t cpuBytes;
        size_t gpuBytes;
    };

	ModelInfo(const torch::nn::Module& model);
	~ModelInfo() = default;

    ModelParams CountParams() const;
    MemoryInfo GetMemorySize() const;

protected:
    const torch::nn::Module& model;
};

#endif