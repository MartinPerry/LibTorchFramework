#include "WarmupCosineScheduler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

WarmupCosineScheduler::WarmupCosineScheduler(
    torch::optim::Optimizer& optimizer,
    int64_t totalSteps,
    double baseLr,
    double warmupPercentage,
    double minLrRatio,
    double warmupMinLrRatio) :
    optimizer(optimizer),
    totalSteps(totalSteps),
    warmupSteps(static_cast<int64_t>(warmupPercentage* totalSteps)),
    baseLr(baseLr),
    minLrRatio(minLrRatio),
    warmupMinLrRatio(warmupMinLrRatio),
    currentStep(0)
{
    if (totalSteps <= 0)
    {
        throw std::invalid_argument("totalSteps must be greater than zero.");
    }

    if (warmupSteps <= 0)
    {
        throw std::invalid_argument("warmupSteps must be greater than zero.");
    }

    // Set the initial learning rate to the warmup minimum.
    const double initialLr = baseLr * warmupMinLrRatio;

    for (auto& group : optimizer.param_groups())
    {        
        group.options().set_lr(initialLr);
    }
}

void WarmupCosineScheduler::Step()
{
    ++currentStep;

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

    for (auto& group : optimizer.param_groups())
    {
        group.options().set_lr(lr);        
    }
}