#ifndef WARMUP_COSINE_SCHEDULER_H
#define WARMUP_COSINE_SCHEDULER_H

#include <torch/torch.h>

#include "./AbstractScheduler.h"

class WarmupCosineScheduler : public AbstractScheduler
{
public:
    WarmupCosineScheduler(
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        int totalSteps,
        double baseLr,
        double warmupPercentage = 0.2,
        double minLrRatio = 1.0e-3,
        double warmupMinLrRatio = 0.0);

    void Step() override;

protected:
    
};

#endif