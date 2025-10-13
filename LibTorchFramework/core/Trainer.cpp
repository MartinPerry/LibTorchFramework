#include "./Trainer.h"

#include "../InputProcessing/DataLoaderData.h"

#include "./Structures.h"

#include "../Settings.h"
#include "../PerformanceSettings.h"

#include "../Utils/ProgressBar.h"

#include "./Metrics/MetricsDefault.h"

#include "./Snapshot/SnapshotSaver.h"

Trainer::Trainer(const Settings& sets, std::shared_ptr<AbstractModel> model) :
	Runner(RunMode::TRAIN, sets, model),
    bestMetrics(nullptr)
{
}

Trainer::~Trainer()
{
}

void Trainer::RunTrainSteps(at::Tensor loss, std::shared_ptr<torch::optim::Optimizer> optimizer)
{
    loss.backward();

    if (optimizer)
    {
        optimizer->step();
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

    torch::GradMode::set_enabled(true);   
    
    //if ((self.scaler is None) and (self.perfSettings.enableAutoCast)) :
    //    self.scaler = torch.cuda.amp.GradScaler()
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

    this->RunTrainSteps(loss, optimizer);

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
