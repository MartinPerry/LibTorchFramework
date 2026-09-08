#ifndef ABSTRACT_SCHEDULER_H
#define ABSTRACT_SCHEDULER_H

#include <torch/torch.h>

class AbstractScheduler
{
public:
    AbstractScheduler(
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        int totalSteps,
        double baseLr,
        double warmupPercentage = 0.2,
        double minLrRatio = 1.0e-3,
        double warmupMinLrRatio = 0.0);

    virtual void Step() = 0;
    
protected:
    std::shared_ptr<torch::optim::Optimizer> optimizer;

    int totalSteps;
    int warmupSteps;

    double baseLr;
    double minLrRatio;
    double warmupMinLrRatio;
    
    int currentStep;
};

#endif