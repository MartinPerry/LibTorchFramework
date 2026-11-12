#ifndef TRAINING_HELPER_H
#define TRAINING_HELPER_H

struct DataLoaderData;
class MetricsDefault;
class DefaultDataset;
class ProgressBar;

#include <memory>
#include <functional>

#include <torch/torch.h>
#include <c10/core/DeviceGuard.h>

#include <Utils/Logger.h>

#include "../core/Structures.h"
#include "../core/AbstractModel.h"

#include "../core/Structures.h"
#include "../core/Runner.h"
#include "../core/Trainer.h"

#include "../core/Distributed/NcclTrainer.h"

#include "../InputProcessing/InputLoader.h"
#include "../InputProcessing/InputLoadersWrapper.h"

#include "../Settings.h"



class TrainingHelper
{
public:
    
    TrainingHelper(const Settings& sets, std::shared_ptr<AbstractModel> model);
    TrainingHelper(const Settings& sets, std::function<std::shared_ptr<AbstractModel>(size_t)> modelIniter,
        int gpuCount);
    ~TrainingHelper() = default;

	template <typename DatasetType = DefaultDataset>
	void Run(std::shared_ptr<InputLoadersWrapper> loaders);

protected:
        
    const Settings& sets;
    std::shared_ptr<AbstractModel> model;

    std::function<std::shared_ptr<AbstractModel>(size_t)> modelIniter;
    int gpuCount;
   
    template <typename DatasetType>
    auto BuildDataLoader(RunMode type, std::shared_ptr<InputLoadersWrapper> loaders, int& bacthesCount);


};


//================================================================================

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
    
    int batchesCountTrain = 0;
    int batchesCountValid = 0;
    int batchesCountTest = 0;

    auto dlTrain = this->BuildDataLoader<DatasetType>(RunMode::TRAIN, loaders, batchesCountTrain);
    auto dlValid = this->BuildDataLoader<DatasetType>(RunMode::VALID, loaders, batchesCountValid);
    auto dlTest = this->BuildDataLoader<DatasetType>(RunMode::TEST, loaders, batchesCountTest);
          
#ifdef LIBTORCH_FRAMEWORK_HAS_NCCL           
    std::shared_ptr<Runner> train = nullptr;
    if (gpuCount > 1)
    {
        std::vector<std::shared_ptr<AbstractModel>> models;
        for (int i = 0; i < gpuCount; i++)
        {
            models.emplace_back(modelIniter(i));
        }

        model = models.front();
        train = std::make_shared<NcclTrainer>(sets, models);
    }
    else
    {
        if (model == nullptr)
        {
            model = modelIniter(0);
        }

        train = std::make_shared<Trainer>(sets, model);
    }
#else
    if (gpuCount > 1)
    {
        MY_LOG_ERROR("NCCL is not available. Will use single GPU training.");
    }

    if (model == nullptr)
    {
        model = modelIniter(0);
    }

    std::shared_ptr<Runner> train = std::make_shared<Trainer>(sets, model);
#endif
        
    if (sets.device == c10::DeviceType::CUDA)
    {
        // Replica factories may leave the last GPU selected. Validation and test
        // use the primary replica and must not move it to that last GPU.
        c10::OptionalDeviceGuard primaryDeviceGuard;
        if (gpuCount > 1)
        {
            primaryDeviceGuard.reset_device(torch::Device(torch::kCUDA, 0));
        }

        MY_LOG_INFO("Running on device: CUDA (gpu count: %d)", gpuCount);
    }
    else
    {
        auto tmp = c10::DeviceTypeName(sets.device);

        MY_LOG_INFO("Running on device: %s", tmp.c_str());
    }

    if (train)
    {
        train->OnStart();
    }

    Runner runnerValid(RunMode::VALID, sets, model);
    Runner runnerTest(RunMode::TEST, sets, model);

    runnerValid.OnStart();
    runnerTest.OnStart();

    if (sets.iterType == Settings::IterationType::EPOCH)
    {
        for (int epochId = 0; epochId < sets.iterCount; epochId++)
        {
            MY_LOG_INFO("Running epoch: %d / %d", epochId, sets.iterCount);

            if (dlTrain)
            {
                MY_LOG_INFO("Batches count (train): %d", batchesCountTrain);
                train->RunEpoch(dlTrain, epochId, batchesCountTrain);
            }

            if (dlValid)
            {
                MY_LOG_INFO("Batches count (valid): %d", batchesCountValid);
                runnerValid.RunEpoch(dlValid, epochId, batchesCountValid);
            }
            else if (dlTest)
            {
                MY_LOG_INFO("Batches count (test): %d", batchesCountTest);
                runnerTest.RunEpoch(dlTest, epochId, batchesCountTest);
            }
        }
    }
    else
    {
        int epochId = 0;
        while (train->GetProcessedBatchesCount() < sets.iterCount)
        {
            MY_LOG_INFO("Running iteration: %d / %d", train->GetProcessedBatchesCount(), sets.iterCount);

            if (dlTrain)
            {
                MY_LOG_INFO("Batches count (train): %d", batchesCountTrain);
                train->RunEpoch(dlTrain, epochId, batchesCountTrain);
            }

            if (dlValid)
            {
                MY_LOG_INFO("Batches count (valid): %d", batchesCountValid);
                runnerValid.RunEpoch(dlValid, epochId, batchesCountValid);
            }
            else if (dlTest)
            {
                MY_LOG_INFO("Batches count (test): %d", batchesCountTest);
                runnerTest.RunEpoch(dlTest, epochId, batchesCountTest);
            }

            epochId++;
        }
    }
    
    if (dlTest)
    {
        MY_LOG_INFO("Batches count (test): %d", batchesCountTest);
        runnerTest.RunEpoch(dlTest, 0, batchesCountTest);
    }

    if (train)
    {
        train->OnEnd();
    }

    runnerValid.OnEnd();
    runnerTest.OnEnd();
}

#endif
