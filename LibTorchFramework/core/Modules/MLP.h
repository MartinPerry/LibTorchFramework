#ifndef MLP_H
#define MLP_H

#include <optional>

#include <torch/torch.h>

class MlpImpl : public torch::nn::Module
{
public:
    MlpImpl(
        int64_t inFeatures,
        std::optional<int64_t> hiddenFeatures = std::nullopt,
        std::optional<int64_t> outFeatures = std::nullopt,
        double drop = 0.0);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear fc1{ nullptr };
    torch::nn::GELU act;
    torch::nn::Linear fc2{ nullptr };
    torch::nn::Dropout drop{ nullptr };
};

TORCH_MODULE(Mlp);

#endif