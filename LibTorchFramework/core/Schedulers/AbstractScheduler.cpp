#include "./AbstractScheduler.h"


AbstractScheduler::AbstractScheduler(
    std::shared_ptr<torch::optim::Optimizer> optimizer,
    int totalSteps,
    double baseLr,
    double warmupPercentage,
    double minLrRatio,
    double warmupMinLrRatio) :
    optimizer(optimizer),
    totalSteps(totalSteps),
    warmupSteps(static_cast<int>(warmupPercentage* totalSteps)),
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

    if (warmupSteps <= 0 || warmupSteps >= totalSteps)
    {
        throw std::invalid_argument("warmupSteps must be greater than zero and less than totalSteps.");
    }

    // Set the initial learning rate to the warmup minimum.
    const double initialLr = baseLr * warmupMinLrRatio;

    for (auto& group : optimizer->param_groups())
    {
        group.options().set_lr(initialLr);
    }
}
