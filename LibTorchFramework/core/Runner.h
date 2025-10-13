#ifndef RUNNER_H
#define RUNNER_H

struct DataLoaderData;
class MetricsDefault;
class DefaultDataset;
class ProgressBar;

#include <torch/torch.h>

#include <Utils/Logger.h>

#include "./Structures.h"

#include "../InputProcessing/InputLoader.h"

#include "../Settings.h"

#include "./AbstractModel.h"

class Runner
{
public:
    Runner(RunMode type, const Settings& sets, std::shared_ptr<AbstractModel> model);
    virtual ~Runner();
    
    template <typename DatasetType = DefaultDataset>
    void Run(std::shared_ptr<InputLoader> loader);

    template <typename DataLoaderType>
    void RunEpoch(DataLoaderType& dl, int epochId, int batchesCount);

protected:
    RunMode type;
    const Settings& sets;

    std::shared_ptr<AbstractModel> model;

    std::shared_ptr<MetricsDefault> metrics;
    
    std::shared_ptr<ProgressBar> pBar;
    
    size_t batchIndex;
    size_t dataLoaderBatchesCount;

    int activeEpochId;

    torch::Tensor ForwardAndLoss(DataLoaderData& batch);

    virtual void OnEpochStart();
    virtual void ProcessBatch(DataLoaderData& batch);
    virtual void OnEpochEnd();
    
};

//================================================================================

template <typename DatasetType>
void Runner::Run(std::shared_ptr<InputLoader> loader)
{
    loader->Load();

    int bacthesCount = 0;
    auto dl = loader->BuildDataLoader<DatasetType>(sets, bacthesCount);
    
    for (int i = 0; i < sets.epochCount; i++)
    {
        MY_LOG_INFO("Running epoch: %d / %d", i, sets.epochCount);

        //self._startTime("epoch")
        this->RunEpoch(dl, i, bacthesCount);
        //self._logTime("epoch")
    }
}

template <typename DataLoaderType>
void Runner::RunEpoch(DataLoaderType& dl, int epochId, int batchesCount)
{
    this->dataLoaderBatchesCount = batchesCount;
    this->activeEpochId = epochId;

    this->OnEpochStart();

    model->to(sets.device);

    model->OnEpochStart();

    batchIndex = 0;
    
    for (auto& batch : *dl)
    {
        model->OnBatchStart();

        batch.setupDevice(sets);

        this->ProcessBatch(batch);

        model->OnBatchEnd();

        batchIndex++;
    }

    model->OnEpochEnd();

    this->OnEpochEnd();
}

#endif