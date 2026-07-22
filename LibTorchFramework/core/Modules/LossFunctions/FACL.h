#ifndef FACL_H
#define FACL_H

#include <torch/torch.h>

class FACLImpl : public torch::nn::Module
{
public:
    FACLImpl(
        int64_t totalStep,
        double constRatio = 0.4,
        double probInit = 1.0,
        double probEnd = 0.0,
        bool includeSigmoid = false);

    torch::Tensor forward(const torch::Tensor& pred, const torch::Tensor& gt);

private:
    double mProbInit;
    double mProbEnd;
    bool mIncludeSigmoid;

    torch::Tensor mProbThresholds;

    int64_t mStep;

    double GetThreshold();

    torch::Tensor FAL(const torch::Tensor& fftPred, const torch::Tensor& fftGt);
    torch::Tensor FCL(const torch::Tensor& fftPred, const torch::Tensor& fftGt);
};

TORCH_MODULE(FACL);

#endif