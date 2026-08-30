#include "./TrainingHelper.h"

TrainingHelper::TrainingHelper(const Settings& sets, std::shared_ptr<AbstractModel> model) :
    sets(sets),
    model(model),
    modelIniter(nullptr),
    gpuCount(1)
{
}

TrainingHelper::TrainingHelper(const Settings& sets, 
    std::function<std::shared_ptr<AbstractModel>(size_t)> modelIniter,
    int gpuCount) :
    sets(sets),
    model(nullptr),
    modelIniter(modelIniter),
    gpuCount(gpuCount)
{
}
