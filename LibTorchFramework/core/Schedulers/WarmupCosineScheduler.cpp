#include "WarmupCosineScheduler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

WarmupCosineScheduler::WarmupCosineScheduler(
    std::shared_ptr<torch::optim::Optimizer> optimizer,
    int totalSteps,
    double baseLr,
    double warmupPercentage,
    double minLrRatio,
    double warmupMinLrRatio) :
    AbstractScheduler(optimizer, totalSteps, baseLr, warmupPercentage, minLrRatio, warmupMinLrRatio)
{
   
}

void WarmupCosineScheduler::Step()
{
    currentStep++;

    double lr = baseLr;

    if (currentStep <= warmupSteps)
    {
        // Equivalent to:
        //
        // warmup_min_lr_ratio + (1 - warmup_min_lr_ratio) * step / warmup_steps
        //
        const double progress = static_cast<double>(currentStep) / static_cast<double>(warmupSteps);
        const double lrRatio = warmupMinLrRatio + (1.0 - warmupMinLrRatio) * progress;

        lr = baseLr * lrRatio;
    }
    else
    {
        const int64_t cosineSteps = totalSteps - warmupSteps;
        const int64_t cosineStep = currentStep - warmupSteps;

        const double progress = std::clamp(
            static_cast<double>(cosineStep) / static_cast<double>(cosineSteps),
            0.0, 
            1.0
        );

        // CosineAnnealingLR:
        //
        // eta = eta_min +
        //       (base_lr - eta_min) *
        //       (1 + cos(pi * progress)) / 2
        const double etaMin = baseLr * minLrRatio;

        lr = etaMin + (baseLr - etaMin) * (1.0 + std::cos(M_PI * progress)) / 2.0;
    }

    for (auto& group : optimizer->param_groups())
    {
        group.options().set_lr(lr);        
    }
}