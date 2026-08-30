#include "./TrainingHelper.h"

TrainingHelper::TrainingHelper(const Settings& sets, std::shared_ptr<AbstractModel> model) :
    TrainingHelper(sets, model, 1)
{
}

TrainingHelper::TrainingHelper(const Settings& sets, std::shared_ptr<AbstractModel> model, int gpuCount) :
    sets(sets),
    model(model),
    gpuCount(gpuCount)
{
}