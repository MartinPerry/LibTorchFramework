#include "./NcclTrainer.h"

#ifdef LIBTORCH_FRAMEWORK_HAS_NCCL

#include "../../InputProcessing/DataLoaderData.h"

#include "../Structures.h"

#include "../../Settings.h"
#include "../../PerformanceSettings.h"

#include "../../Utils/ProgressBar.h"

#include "../Metrics/MetricsDefault.h"

#include "../Snapshot/SnapshotSaver.h"

#include "../Modules/gradscaler.hpp"

#include "../CudaGraphHelper.h"

#include "./NcclTrainerContext.h"

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <string_view>
#include <utility>


//debug with single GPU
bool IsSingleGpuNcclTestEnabled()
{
    return true;    
}


NcclTrainer::NcclTrainer(const Settings& sets, std::vector<std::shared_ptr<AbstractModel>> models) :
    Runner(RunMode::TRAIN, sets, models.front()),
    cudaGraph(nullptr),
    replicaModels(std::move(models)),
    bestMetrics(nullptr),
    nccl(nullptr),
    distributedParametersSynchronized(false)
{
    for (size_t i = 0; i < replicaModels.size(); i++)
    {
        TORCH_CHECK(replicaModels[i] != nullptr, "Trainer model ", i, " is null");
        for (size_t j = 0; j < i; j++)
        {
            TORCH_CHECK(
                replicaModels[i].get() != replicaModels[j].get(),
                "Each GPU needs a distinct model instance");
        }
    }

    const bool singleGpuNcclTest = (replicaModels.size() == 1) && IsSingleGpuNcclTestEnabled();
    if ((replicaModels.size() > 1) || singleGpuNcclTest)
    {
        TORCH_CHECK(
            sets.device == torch::kCUDA,
            "NCCL training requires a CUDA device");
        nccl = std::make_shared<NcclTrainerContext>(replicaModels.size());
        if (singleGpuNcclTest)
        {
            MY_LOG_INFO(
                "NCCL single-GPU smoke mode enabled; collectives use one rank");
        }
    }

    scalers.reserve(replicaModels.size());
    for (size_t i = 0; i < replicaModels.size(); i++)
    {
        if (sets.perf.enableAutoCast)
        {
            scalers.push_back(std::make_shared<torch::amp::GradScaler>());
        }
        else
        {
            scalers.push_back(nullptr);
        }
    }

    //cudaGraph = std::make_shared<CudaGraphHelper>(this, 1, true, true);
}

NcclTrainer::~NcclTrainer()
{
}

void NcclTrainer::SelectCudaDevice(size_t device)
{
    const auto result = cudaSetDevice(static_cast<int>(device));
    TORCH_CHECK(
        result == cudaSuccess,
        "cudaSetDevice failed: ", cudaGetErrorString(result));
}

void NcclTrainer::CheckLoss(
    at::Tensor loss,
    const std::shared_ptr<AbstractModel>& activeModel)
{
    TORCH_CHECK(loss.defined(), "Loss is not defined (autocast)");
    TORCH_CHECK(loss.grad_fn() != nullptr, "Loss has no grad_fn (autocast)");
    TORCH_CHECK(loss.sizes().size() == 0, "Loss is not scalar; sizes = ", loss.sizes());

    auto params = activeModel->parameters();
    TORCH_CHECK(!params.empty(), "Model has no parameters");
    TORCH_CHECK(
        loss.device() == params.front().device(),
        "Device mismatch: loss on ", loss.device(),
        " but model params on ", params.front().device());
    TORCH_CHECK(
        loss.dtype() == torch::kFloat32 ||
        loss.dtype() == torch::kFloat16 ||
        loss.dtype() == torch::kBFloat16,
        "Unexpected loss dtype: ", loss.dtype());
}

std::vector<std::vector<torch::Tensor>> NcclTrainer::ParametersByDevice() const
{
    std::vector<std::vector<torch::Tensor>> result;
    result.reserve(replicaModels.size());
    for (const auto& replica : replicaModels)
    {
        result.push_back(replica->parameters());
    }
    return result;
}

std::vector<DataLoaderData> NcclTrainer::BuildReplicaBatches(
    DataLoaderData& batch) const
{
    if (replicaModels.size() == 1)
    {
        return { batch };
    }

    const size_t totalBatchSize = batch.GetBatchSize();
    TORCH_CHECK(totalBatchSize > 0, "Cannot train on an empty dataloader batch");
    TORCH_CHECK(
        batch.input.dim() > 0 &&
        static_cast<size_t>(batch.input.size(0)) == totalBatchSize,
        "Input tensor batch dimension does not match DataLoaderData indices");
    TORCH_CHECK(
        batch.target.dim() > 0 &&
        static_cast<size_t>(batch.target.size(0)) == totalBatchSize,
        "Target tensor batch dimension does not match DataLoaderData indices");

    const auto& allIndices = batch.GetDataIndices();
    std::vector<DataLoaderData> batches;
    batches.reserve(replicaModels.size());

    size_t offset = 0;
    for (size_t device = 0; device < replicaModels.size(); device++)
    {
        const size_t count = totalBatchSize / replicaModels.size() +
            (device < (totalBatchSize % replicaModels.size()) ? 1 : 0);
        const size_t end = offset + count;

        std::vector<int64_t> indices(
            allIndices.begin() + static_cast<std::ptrdiff_t>(offset),
            allIndices.begin() + static_cast<std::ptrdiff_t>(end)
        );

        DataLoaderData replicaBatch(std::move(indices));
        const torch::Device targetDevice(torch::kCUDA, static_cast<int8_t>(device));

        replicaBatch.input = batch.input
            .slice(0, static_cast<int64_t>(offset), static_cast<int64_t>(end))
            .to(targetDevice, batch.input.dtype(), sets.perf.useNonBlockingTransfers);

        replicaBatch.target = batch.target
            .slice(0, static_cast<int64_t>(offset), static_cast<int64_t>(end))
            .to(targetDevice, batch.target.dtype(), sets.perf.useNonBlockingTransfers);

        for (const auto& [name, value] : batch.additionalData)
        {
            TORCH_CHECK(
                value.dim() > 0 && static_cast<size_t>(value.size(0)) == totalBatchSize,
                "Additional tensor '", name, "' has no matching batch dimension");
            
            replicaBatch.additionalData.emplace(
                name,
                value.slice(0, static_cast<int64_t>(offset), static_cast<int64_t>(end))
                    .to(targetDevice, value.dtype(), sets.perf.useNonBlockingTransfers)
            );
        }

        batches.push_back(std::move(replicaBatch));
        offset = end;
    }
    return batches;
}

void NcclTrainer::RunTrainStepsFull(
    std::vector<torch::Tensor>& losses,
    bool canUpdate)
{
    for (size_t device = 0; device < losses.size(); device++)
    {
#ifdef _DEBUG
        CheckLoss(losses[device], replicaModels[device]);
#endif
        SelectCudaDevice(device);
        losses[device].backward();
    }

    if (canUpdate)
    {
        RunOptimizerFull();
    }
}

void NcclTrainer::RunOptimizerFull()
{
    nccl->AllReduceGradients(ParametersByDevice());
    

    for (size_t device = 0; device < replicaModels.size(); device++)
    {
        auto optimizer = replicaModels[device]->optimizer;
        TORCH_CHECK(optimizer != nullptr, "Model replica ", device, " has no optimizer");
        SelectCudaDevice(device);
        if (sets.clippingFn)
        {
            sets.clippingFn(replicaModels[device]->parameters());
        }
        optimizer->step();
        optimizer->zero_grad();
    }
}

void NcclTrainer::RunTrainStepsAutocast(
    std::vector<torch::Tensor>& losses,
    bool canUpdate)
{
    for (size_t device = 0; device < losses.size(); device++)
    {
#ifdef _DEBUG
        CheckLoss(losses[device], replicaModels[device]);
#endif
        SelectCudaDevice(device);
        scalers[device]->scale(losses[device]).backward();
    }

    if (canUpdate)
    {
        RunOptimizerAutoCast();
    }
}

void NcclTrainer::RunOptimizerAutoCast()
{
    
    bool globalNonFinite = nccl->HasGlobalNonFiniteGradients(ParametersByDevice());
    if (globalNonFinite)
    {
        nccl->MarkGradientsNonFinite(ParametersByDevice());
    }

    for (size_t device = 0; device < replicaModels.size(); device++)
    {
        SelectCudaDevice(device);
        scalers[device]->unscale_(*replicaModels[device]->optimizer);
    }

    if (!globalNonFinite)
    {
        nccl->AllReduceGradients(ParametersByDevice());
    }
    
    for (size_t device = 0; device < replicaModels.size(); device++)
    {
        auto optimizer = replicaModels[device]->optimizer;
        TORCH_CHECK(optimizer != nullptr, "Model replica ", device, " has no optimizer");
        SelectCudaDevice(device);

        if (sets.clippingFn && !globalNonFinite)
        {            
            sets.clippingFn(replicaModels[device]->parameters());
        }

        scalers[device]->step(*optimizer);
        scalers[device]->update();
        optimizer->zero_grad();
    }
}

void NcclTrainer::RunStep(DataLoaderData& batch, bool canUpdate)
{
    auto replicaBatches = BuildReplicaBatches(batch);
    const size_t totalBatchSize = batch.GetBatchSize();
    std::vector<torch::Tensor> losses;
    losses.reserve(replicaModels.size());

    for (size_t device = 0; device < replicaModels.size(); device++)
    {
        if (device > 0)
        {
            replicaModels[device]->OnBatchStart();
        }
        SelectCudaDevice(device);
       
        if (replicaBatches[device].GetBatchSize() == 0)
        {
            // Keep every communicator participating when a short final batch
            // has fewer samples than GPUs. This rank contributes zero.

            torch::Tensor zeroReplicaLoss = {};
            
            for (const auto& parameter : replicaModels[device]->parameters())
            {
                if (parameter.requires_grad() && (parameter.numel() > 0))
                {
                    zeroReplicaLoss = parameter.reshape({ -1 }).select(0, 0) * 0.0;                    
                    break;
                }
                TORCH_CHECK(false, "Model replica ", device, " has no trainable parameters");                
            }

            losses.push_back(zeroReplicaLoss);
        }
        else
        {
            // NCCL averages replica gradients. Weight uneven shards so that the
            // result still equals the gradient of the complete input batch.
            const double shardWeight = replicaModels.size() == 1
                ? 1.0
                : static_cast<double>(replicaModels.size()) *
                static_cast<double>(replicaBatches[device].GetBatchSize()) /
                static_cast<double>(totalBatchSize);

            auto loss = this->ForwardAndLoss(replicaBatches[device], replicaModels[device]);
            loss *= shardWeight;            
        }
    }

    if (sets.perf.enableAutoCast)
    {
        RunTrainStepsAutocast(losses, canUpdate);
    }
    else
    {
        RunTrainStepsFull(losses, canUpdate);
    }

    float averageLoss = 0.0f;
    for (const auto& loss : losses)
    {
        averageLoss += loss.item().toFloat() / static_cast<float>(losses.size());
    }
    ProgressLoss(averageLoss);

    for (size_t device = 1; device < replicaModels.size(); device++)
    {
        replicaModels[device]->OnBatchEnd();
    }
}

torch::Tensor NcclTrainer::ForwardAndLoss(DataLoaderData& batch, std::shared_ptr<AbstractModel> model)
{
    const auto& tmp = this->model;

    this->model = model;
    auto loss = Runner::ForwardAndLoss(batch);
    this->model = tmp;

    return loss;
}

void NcclTrainer::ProgressLoss(float loss)
{
    this->pBar->SetParam("loss", std::to_string(loss));
    this->pBar->NextStep();
}

void NcclTrainer::PrepareBatch(DataLoaderData& batch)
{
    if (replicaModels.size() == 1)
    {
        Runner::PrepareBatch(batch);
    }
    // Multi-GPU batches remain in pinned host memory and are split before
    // being copied directly to their destination devices.
}

void NcclTrainer::PrepareModel()
{
    if (replicaModels.size() == 1)
    {
        Runner::PrepareModel();
        return;
    }

    for (size_t device = 0; device < replicaModels.size(); device++)
    {
        SelectCudaDevice(device);
        replicaModels[device]->to(
            torch::Device(torch::kCUDA, static_cast<int8_t>(device)));
    }
}

void NcclTrainer::OnEpochStart()
{
    Runner::OnEpochStart();

    for (size_t device = 0; device < replicaModels.size(); device++)
    {
        SelectCudaDevice(device);
        replicaModels[device]->train();
    }

    torch::autograd::GradMode::set_enabled(true);
}

void NcclTrainer::OnModelEpochStart()
{
    for (size_t device = 1; device < replicaModels.size(); device++)
    {
        SelectCudaDevice(device);
        replicaModels[device]->OnEpochStart();
    }

    if (!distributedParametersSynchronized)
    {
        nccl->Broadcast(ParametersByDevice());
        distributedParametersSynchronized = true;
    }

    std::vector<std::vector<torch::Tensor>> buffersByDevice;
    buffersByDevice.reserve(replicaModels.size());
    for (const auto& replica : replicaModels)
    {
        buffersByDevice.push_back(replica->buffers());
    }
    nccl->Broadcast(buffersByDevice);    
}

void NcclTrainer::ProcessBatch(DataLoaderData& batch)
{
    bool canUpdate = true;
    if (sets.gradientAccumulationCount.has_value() && (*sets.gradientAccumulationCount > 0))
    {
        canUpdate = ((batchIndex + 1) % *sets.gradientAccumulationCount == 0) ||
            (batchIndex + 1 == dataLoaderBatchesCount);
    }

    if (canUpdate)
    {
        for (size_t device = 0; device < replicaModels.size(); device++)
        {
            if (replicaModels[device]->optimizer == nullptr)
            {
                MY_LOG_WARNING("Model replica %zu has no optimizer", device);
                canUpdate = false;
            }
        }
    }

    TORCH_CHECK(
        !cudaGraph || (replicaModels.size() == 1),
        "CUDA graph training is not supported with in-process NCCL replicas");
    if (cudaGraph)
    {
#ifdef USE_CUDA
        cudaGraph->Run(batch, canUpdate ? model->optimizer : nullptr);
#endif
    }
    else
    {
        RunStep(batch, canUpdate);
    }
}

void NcclTrainer::OnEpochEnd()
{
    for (size_t device = 1; device < replicaModels.size(); device++)
    {
        replicaModels[device]->OnEpochEnd();
    }

    Runner::OnEpochEnd();

    if ((this->metrics) && (this->metrics->IsBetterThan(this->bestMetrics)))
    {
        SnapshotSaver saver(this->model.get());
        saver.Save(sets.pretrainedManager);
        this->bestMetrics = this->metrics;
    }
}

#endif