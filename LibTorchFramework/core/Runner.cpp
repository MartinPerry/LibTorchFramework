#include "./Runner.h"

#include <ATen/autocast_mode.h>

#include "../InputProcessing/DataLoaderData.h"

#include "./Metrics/MetricsDefault.h"

#include "./Snapshot/PretrainedManager.h"

#include "../Settings.h"
#include "../PerformanceSettings.h"

#include "../Utils/ProgressBar.h"

Runner::Runner(RunMode type, const Settings& sets, std::shared_ptr<AbstractModel> model) :
    type(type),
	sets(sets),
	model(model),
    metrics(nullptr),
    activeEpochId(0)
{
    this->pBar = std::make_shared<ProgressBar>();
}

Runner::~Runner()
{
}

torch::Tensor Runner::ForwardAndLoss(DataLoaderData& batch)
{    
    
    if (sets.perf.enableAutoCast)
    {                        
        at::autocast::set_autocast_enabled(sets.device, true);
        if (sets.perf.autocastType.has_value())
        {
            at::autocast::set_autocast_dtype(sets.device, *sets.perf.autocastType);
        }
    }
    auto result = model->RunForward(batch);
  
    torch::Tensor loss;

    if (sets.lossFn)
    {
        loss = sets.lossFn(result, batch.target);
        
        if ((sets.gradientAccumulationCount.has_value()) and (*sets.gradientAccumulationCount > 0))
        {
            loss = loss / *sets.gradientAccumulationCount;
        }
    }

    if (sets.perf.enableAutoCast)
    {
        at::autocast::clear_cache();
        at::autocast::set_autocast_enabled(sets.device, false);        
    }

    if (this->metrics)
    {
        //???
        //outputs = outputs.detach()

        this->metrics->UpdateProcessCounter();
        if (this->metrics->CanProcess())
        {
            this->metrics->AddLoss(loss);
            //metric.addAdditionalLoss(f"loss{i}", losses[i])

            this->metrics->AddDataIndices(batch.GetDataIndices());
            this->metrics->AddPredictionTarget(result[0], batch.target);
        }
    }


    return loss;
}

//============================================================
// Main loop callbacks
//============================================================

void Runner::OnEpochStart()
{
    if (sets.metricsInitFn)
    {
        this->metrics = sets.metricsInitFn();
    }

    model->eval();

    torch::autograd::GradMode::set_enabled(false);

    this->pBar->ClearParams();
    this->pBar->Start(this->dataLoaderBatchesCount);
}

void Runner::ProcessBatch(DataLoaderData& batch)
{
    this->pBar->NextStep();

    /*
    auto bData = batch.input;
    auto bTarget = batch.target;

    std::cout << "Runner" << std::endl;
    std::cout << "input: " << bData << std::endl;
    std::cout << "target: " << bTarget << std::endl;

    printf("xx");
    */
}

void Runner::OnEpochEnd()
{
    if ((sets.pretrainedManager) && (this->metrics))
    {
        std::string runType = "";
        if (type == RunMode::TRAIN)
        {
            runType = "train";
        }
        else if (type == RunMode::VALID)
        {
            runType = "valid";
        }
        else if (type == RunMode::TEST)
        {
            runType = "test";
        }

        auto path = sets.pretrainedManager->BuildFilePathForSave(this->model.get(), 
            runType, 
            "json", 
            this->activeEpochId, 
            runType);

        this->metrics->Save(path);
    }

    this->pBar->Finish();
}