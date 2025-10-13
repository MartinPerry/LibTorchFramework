#ifndef TRAINING_HELPER_H
#define TRAINING_HELPER_H

struct DataLoaderData;
class MetricsDefault;
class DefaultDataset;
class ProgressBar;

#include <memory>

#include <torch/torch.h>

#include <Utils/Logger.h>

#include "../core/Structures.h"
#include "../core/AbstractModel.h"

#include "../core/Structures.h"
#include "../core/Runner.h"
#include "../core/Trainer.h"

#include "../InputProcessing/InputLoader.h"
#include "../InputProcessing/InputLoadersWrapper.h"

#include "../Settings.h"



class TrainingHelper
{
public:
    TrainingHelper(const Settings& sets, std::shared_ptr<AbstractModel> model);
    ~TrainingHelper() = default;

	template <typename DatasetType = DefaultDataset>
	void Run(std::shared_ptr<InputLoadersWrapper> loaders);

protected:
    
    const Settings& sets;
    std::shared_ptr<AbstractModel> model;

    template <typename DatasetType>
    auto BuildDataLoader(RunMode type, std::shared_ptr<InputLoadersWrapper> loaders, int& bacthesCount);


};


//================================================================================

TrainingHelper::TrainingHelper(const Settings& sets, std::shared_ptr<AbstractModel> model) : 
    sets(sets),
    model(model)
{
}

template <typename DatasetType>
auto TrainingHelper::BuildDataLoader(RunMode type, 
    std::shared_ptr<InputLoadersWrapper> loaders, int& bacthesCount)
{    
    auto loader = loaders->GetLoader(type);
    if (loader == nullptr)
    {        
        bacthesCount = 0;      
        return InputLoader::StackedDataLoaderType<DatasetType>{};        
    }
    loader->Load();

    return loader->BuildDataLoader<DatasetType>(sets, bacthesCount);
}

template <typename DatasetType>
void TrainingHelper::Run(std::shared_ptr<InputLoadersWrapper> loaders)
{
    int bacthesCountTrain = 0;
    int bacthesCountValid = 0;
    int bacthesCountTest = 0;

    auto dlTrain = this->BuildDataLoader<DatasetType>(RunMode::TRAIN, loaders, bacthesCountTrain);
    auto dlValid = this->BuildDataLoader<DatasetType>(RunMode::VALID, loaders, bacthesCountValid);
    auto dlTest = this->BuildDataLoader<DatasetType>(RunMode::TEST, loaders, bacthesCountTest);
      
    Runner runnerValid(RunMode::VALID, sets, model);
    Runner runnerTest(RunMode::TEST, sets, model);
    Trainer train(sets, model);

    for (int i = 0; i < sets.epochCount; i++)
    {
        MY_LOG_INFO("Running epoch: %d / %d", i, sets.epochCount);
         
        if (dlTrain)
        {
            MY_LOG_INFO("Batches count (train): %d", bacthesCountTrain);
            train.RunEpoch(dlTrain, i, bacthesCountTrain);
        }
        
        if (dlValid)
        {
            MY_LOG_INFO("Batches count (valid): %d", bacthesCountValid);
            runnerValid.RunEpoch(dlValid, i, bacthesCountValid);
        }
        else if (dlTest)
        {   
            MY_LOG_INFO("Batches count (test): %d", bacthesCountTest);
            runnerTest.RunEpoch(dlTest, i, bacthesCountTest);                        
        }
    }
    
    if (dlTest)
    {
        MY_LOG_INFO("Batches count (test): %d", bacthesCountTest);
        runnerTest.RunEpoch(dlTest, 0, bacthesCountTest);
    }
}

#endif
