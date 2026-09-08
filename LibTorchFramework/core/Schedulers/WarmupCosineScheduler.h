#ifndef WARMUP_COSINE_SCHEDULER_H
#define WARMUP_COSINE_SCHEDULER_H

#include <torch/torch.h>

class WarmupCosineScheduler
{
public:
    WarmupCosineScheduler(
        torch::optim::Optimizer& optimizer,
        int64_t totalSteps,
        double baseLr,
        double warmupPercentage = 0.2,
        double minLrRatio = 1.0e-3,
        double warmupMinLrRatio = 0.0);

    void Step();

    torch::optim::LRScheduler& GetScheduler();

private:
    torch::optim::Optimizer& optimizer;

    int64_t totalSteps;
    int64_t warmupSteps;

    double baseLr;
    double minLrRatio;
    double warmupMinLrRatio;

    // Stub:
    // LibTorch does not provide a direct equivalent of Python's
    // SequentialLR + LambdaLR combination with an arbitrary lambda.
    // Implement the scheduling logic directly in Step().
    int64_t currentStep;
};

#endif