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
    
    size_t GetProcessedBatchesCount() const;

    void OnStart();
    void OnEnd();

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
    
    size_t batchIndex; //index of batch within active epoch
    size_t dataLoaderBatchesCount; //number of batches loaded from dataLoader for current run
    size_t processedBatchesCount; //total number of processed bateches across all epochs

    int activeEpochId; //active epoch ID

    torch::Tensor ForwardAndLoss(DataLoaderData& batch);    
    torch::Tensor ForwardAndLoss(DataLoaderData& batch,
        const std::shared_ptr<AbstractModel>& activeModel, torch::Tensor* prediction);

    void UpdateMetrics(DataLoaderData& batch, torch::Tensor loss, torch::Tensor prediction);

    virtual void PrepareModel();
    virtual void OnEpochStart();
    virtual void OnModelEpochStart();
    virtual void OnModelBatchStart();
    virtual void OnModelBatchEnd();
    virtual void PrepareBatch(DataLoaderData& batch);
    virtual void ProcessBatch(DataLoaderData& batch);
    virtual void OnModelEpochEnd();
    virtual void OnEpochEnd();
    
};

//================================================================================

template <typename DatasetType>
void Runner::Run(std::shared_ptr<InputLoader> loader)
{
    loader->Load();

    this->OnStart();

    int bacthesCount = 0;
    auto dl = loader->BuildDataLoader<DatasetType>(sets, bacthesCount);
        
    if (sets.iterType == Settings::IterationType::EPOCH)
    {
        for (int epochId = 0; epochId < sets.iterCount; epochId++)
        {
            MY_LOG_INFO("Running epoch: %d / %d", epochId, sets.iterCount);

            this->RunEpoch(dl, epochId, bacthesCount);            
        }
    }
    else 
    {
        int epochId = 0;
        while (this->processedBatchesCount < sets.iterCount)
        {
            MY_LOG_INFO("Running iteration: %d / %d", this->processedBatchesCount, sets.iterCount);

            this->RunEpoch(dl, epochId, bacthesCount);

            epochId++;
        }
    }

    this->OnEnd();
}

template <typename DataLoaderType>
void Runner::RunEpoch(DataLoaderType& dl, int epochId, int batchesCount)
{
    if (sets.iterType == Settings::IterationType::STEPS)
    {
        if (this->processedBatchesCount > sets.iterCount)
        {
            return;
        }
    }

    this->dataLoaderBatchesCount = batchesCount;
    this->activeEpochId = epochId;

    this->PrepareModel();

    this->OnEpochStart();
    
    this->OnModelEpochStart();
    
    batchIndex = 0;
    
    for (auto& batch : *dl)
    {
        this->OnModelBatchStart();

        this->PrepareBatch(batch);

        this->ProcessBatch(batch);

        this->OnModelBatchEnd();

        batchIndex++;
        this->processedBatchesCount++;

        if (sets.iterType == Settings::IterationType::STEPS)
        {
            if (this->processedBatchesCount > sets.iterCount)
            {
                break;
            }
        }
    }

    this->OnModelEpochEnd();

    this->OnEpochEnd();
}

#endif