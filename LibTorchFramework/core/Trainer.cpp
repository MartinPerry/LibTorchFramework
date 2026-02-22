#include "./Trainer.h"

#include "../InputProcessing/DataLoaderData.h"

#include "./Structures.h"

#include "../Settings.h"
#include "../PerformanceSettings.h"

#include "../Utils/ProgressBar.h"

#include "./Metrics/MetricsDefault.h"

#include "./Snapshot/SnapshotSaver.h"

#include "./Modules/gradscaler.hpp"

Trainer::Trainer(const Settings& sets, std::shared_ptr<AbstractModel> model) :
	Runner(RunMode::TRAIN, sets, model),
    scaler(nullptr),
    bestMetrics(nullptr)
{
    if (sets.perf.enableAutoCast)
    {
        scaler = std::make_shared<torch::amp::GradScaler>();        
    }
}

Trainer::~Trainer()
{
}

void Trainer::CheckLoss(at::Tensor loss)
{
    TORCH_CHECK(scaler != nullptr, "enableAutoCast is true but scaler is null");
    TORCH_CHECK(loss.defined(), "Loss is not defined (autocast)");
    TORCH_CHECK(loss.grad_fn() != nullptr, "Loss has no grad_fn (autocast)");

    // Optional, but useful: loss should be scalar (0-D) or you must pass grad_outputs
    TORCH_CHECK(loss.sizes().size() == 0, "Loss is not scalar; sizes = ", loss.sizes());

    // (1) Check devices
    auto params = model->parameters();
    TORCH_CHECK(!params.empty(), "Model has no parameters");

    const auto loss_device = loss.device();
    const auto param_device = params.front().device();
    TORCH_CHECK(loss_device == param_device,
        "Device mismatch: loss on ", loss_device,
        " but model params on ", param_device);

    // (2) Check dtype
    TORCH_CHECK(loss.dtype() == torch::kFloat32 ||
        loss.dtype() == torch::kFloat16 ||
        loss.dtype() == torch::kBFloat16,
        "Unexpected loss dtype: ", loss.dtype());
}

void Trainer::RunTrainStepsFull(at::Tensor loss, std::shared_ptr<torch::optim::Optimizer> optimizer)
{
#ifdef _DEBUG
    this->CheckLoss(loss);
#endif
    loss.backward();

    if (sets.clippingFn)
    {        
        sets.clippingFn(model->parameters());
    }

    if (optimizer)
    {
        optimizer->step();
        optimizer->zero_grad();
    }
}

void Trainer::RunTrainStepsAutocast(at::Tensor loss, std::shared_ptr<torch::optim::Optimizer> optimizer)
{
#ifdef _DEBUG
    this->CheckLoss(loss);
#endif

    auto scaledLoss = scaler->scale(loss);
    scaledLoss.backward();
   
    if (sets.clippingFn)
    {
        if (optimizer)
        {
            scaler->unscale_(*optimizer);
        }
        sets.clippingFn(model->parameters());
    }

    if (optimizer)
    {
        scaler->step(*optimizer);
        scaler->update();

        optimizer->zero_grad();
    }
}

//============================================================
// Main loop callbacks
//============================================================

void Trainer::OnEpochStart()
{
    Runner::OnEpochStart();

    model->train();
    
    torch::autograd::GradMode::set_enabled(true);
}

void Trainer::ProcessBatch(DataLoaderData& batch)
{
    auto loss = this->ForwardAndLoss(batch);

    auto optimizer = this->model->optimizer;
    if ((sets.gradientAccumulationCount.has_value()) and (*sets.gradientAccumulationCount > 0))
    {
        bool canUpdate = ((batchIndex + 1) % *sets.gradientAccumulationCount == 0) || 
            (batchIndex + 1 == dataLoaderBatchesCount);

        if (canUpdate == false)
        {
            optimizer = nullptr;
        }
    }
    else if (optimizer == nullptr)
    {
        MY_LOG_WARNING("No optimizer is set. Model wont train");
    }

    if (sets.perf.enableAutoCast)
    {
        this->RunTrainStepsAutocast(loss, optimizer);
    }
    else
    {
        this->RunTrainStepsFull(loss, optimizer);
    }

    this->pBar->SetParam("loss", std::to_string(loss.item().toFloat()));
    this->pBar->NextStep();
}

void Trainer::OnEpochEnd()
{        
    Runner::OnEpochEnd();

    if ((this->metrics) && (this->metrics->IsBetterThan(this->bestMetrics)))
    {
        SnapshotSaver saver(this->model.get());
        saver.Save(sets.pretrainedManager);

        this->bestMetrics = this->metrics;
    }    
}
