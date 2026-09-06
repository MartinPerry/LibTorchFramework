#include "./Runner.h"

#include <ATen/autocast_mode.h>

#include "../InputProcessing/DataLoaderData.h"

#include "./Metrics/MetricsDefault.h"

#include "./Snapshot/PretrainedManager.h"

#include "../Settings.h"
#include "../PerformanceSettings.h"

#include "../Utils/ProgressBar.h"

namespace
{
    /// <summary>
    /// Scped type will automatically be released when blocks end its lifetime
    /// </summary>
    class ScopedAutocast
    {
    public:
        explicit ScopedAutocast(const Settings& sets) :
            device(sets.device),
            previousEnabled(at::autocast::is_autocast_enabled(device)),
            previousType(at::autocast::get_autocast_dtype(device))
        {
            at::autocast::set_autocast_enabled(device, sets.perf.enableAutoCast);
            if (sets.perf.enableAutoCast && sets.perf.autocastType.has_value())
            {
                at::autocast::set_autocast_dtype(device, *sets.perf.autocastType);
            }
            at::autocast::increment_nesting();
        }

        ~ScopedAutocast()
        {
            if (at::autocast::decrement_nesting() == 0)
            {
                at::autocast::clear_cache();
            }
            at::autocast::set_autocast_enabled(device, previousEnabled);
            at::autocast::set_autocast_dtype(device, previousType);
        }

    private:
        c10::DeviceType device;
        bool previousEnabled;
        at::ScalarType previousType;
    };
}



Runner::Runner(RunMode type, const Settings& sets, std::shared_ptr<AbstractModel> model) :
    type(type),
	sets(sets),
	model(model),
    metrics(nullptr),
    batchIndex(0),
    dataLoaderBatchesCount(0),
    activeEpochId(0)
{
    this->pBar = std::make_shared<ProgressBar>();
}

Runner::~Runner()
{
}

torch::Tensor Runner::ForwardAndLoss(DataLoaderData& batch)
{
    if (this->metrics)
    {
        torch::Tensor prediction;
        auto loss = ForwardAndLoss(batch, model, &prediction);

        this->UpdateMetrics(batch, loss, prediction);

        return loss;
    }
    else 
    {
        return ForwardAndLoss(batch, model, nullptr);
    }    
}

torch::Tensor Runner::ForwardAndLoss(DataLoaderData& batch,
    const std::shared_ptr<AbstractModel>& activeModel, torch::Tensor* prediction)
{
    ScopedAutocast autocast(sets);
    auto result = this->model->RunForward(batch);
  
    torch::Tensor loss;

    if (sets.lossFn)
    {
        loss = sets.lossFn(result, batch.target);
        
        if ((sets.gradientAccumulationCount.has_value()) and (*sets.gradientAccumulationCount > 0))
        {
            loss = loss / *sets.gradientAccumulationCount;
        }
    }

    if (prediction)
    {
        *prediction = result[0].detach();
    }

    
    return loss;
}

void Runner::UpdateMetrics(DataLoaderData& batch, torch::Tensor loss, torch::Tensor prediction)
{
    if (this->metrics == nullptr)
    {
        return;
    }
        
    this->metrics->UpdateProcessCounter();
    if (this->metrics->CanProcess())
    {
        this->metrics->AddLoss(loss);

        this->metrics->AddDataIndices(batch.GetDataIndices());
        this->metrics->AddPredictionTarget(prediction, batch.target);
    }    
}


//============================================================
// Main loop callbacks
//============================================================


void Runner::PrepareModel()
{
    model->to(sets.device);
}

void Runner::OnEpochStart()
{
    if (sets.metricsInitFn)
    {
        this->metrics = sets.metricsInitFn();
        if (this->metrics)
        {
            this->metrics->Reset();
        }
    }

    model->eval();

    torch::autograd::GradMode::set_enabled(false);

    this->pBar->ClearParams();
    this->pBar->Start(this->dataLoaderBatchesCount);
}

void Runner::OnModelEpochStart()
{
    model->OnEpochStart();
}

void Runner::OnModelBatchStart()
{
    model->OnBatchStart();
}

void Runner::OnModelBatchEnd()
{
    model->OnBatchEnd();
}

void Runner::PrepareBatch(DataLoaderData& batch)
{
    batch.setupDevice(sets);
}

void Runner::ProcessBatch(DataLoaderData& batch)
{
    auto loss = this->ForwardAndLoss(batch);

    float fLoss = loss.item().toFloat();

    this->pBar->SetParam("loss", std::to_string(fLoss));
    this->pBar->NextStep();
   
}

void Runner::OnModelEpochEnd()
{
    model->OnEpochEnd();
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

        MetricsDefault::SaveInfo si;
        si.jsonFilePath = path;
        si.epochId = this->activeEpochId;
        si.runMode = type;

        this->metrics->Save(si);
    }

    this->pBar->Finish();
}