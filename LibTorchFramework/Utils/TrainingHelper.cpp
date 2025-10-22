#include "./TrainingHelper.h"

TrainingHelper::TrainingHelper(const Settings& sets, std::shared_ptr<AbstractModel> model) :
    sets(sets),
    model(model)
{
}