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

#include "../Schedulers/AbstractScheduler.h"

#include "../CudaGraphHelper.h"

#include "./NcclTrainerContext.h"
#include "./ReplicaWorkers.h"

#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <utility>


NcclTrainer::NcclTrainer(const Settings& sets, std::vector<std::shared_ptr<AbstractModel>> models) :
    Runner(RunMode::TRAIN, sets, models.front()),    
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

    TORCH_CHECK(sets.device == torch::kCUDA, "NCCL training requires a CUDA device");
    nccl = std::make_shared<NcclTrainerContext>(replicaModels.size());

    scalers.reserve(replicaModels.size());
    for (size_t i = 0; i < replicaModels.size(); i++)
    {
        const c10::cuda::CUDAGuard deviceGuard(static_cast<c10::DeviceIndex>(i));
        if (sets.perf.enableAutoCast)
        {
            scalers.push_back(std::make_shared<torch::amp::GradScaler>());
        }
        else
        {
            scalers.push_back(nullptr);
        }
    }

    workers = std::make_unique<ReplicaWorkers>(replicaModels.size());
    
}

NcclTrainer::~NcclTrainer()
{
}

void NcclTrainer::RunOnReplicas(const std::function<void(size_t)>& fn)
{
    std::vector<c10::cuda::CUDAStream> streams;
    streams.reserve(replicaModels.size());
    for (size_t device = 0; device < replicaModels.size(); ++device)
    {
        streams.push_back(c10::cuda::getCurrentCUDAStream(
            static_cast<c10::DeviceIndex>(device)));
    }

    workers->Run([&](size_t device)
    {
        // Use the same stream as copies and NCCL on the coordinator. Waiting
        // for host tasks ensures submission order; CUDA orders the GPU work.
        // Also inherit source-device streams for cross-device tensor copies.
        const c10::cuda::CUDAMultiStreamGuard streamGuards(streams);
        const c10::cuda::CUDAGuard deviceGuard(static_cast<c10::DeviceIndex>(device));
        const torch::autograd::AutoGradMode gradMode(true);
        fn(device);
    });
}

void NcclTrainer::CheckLoss(at::Tensor loss, const std::shared_ptr<AbstractModel>& activeModel)
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

std::vector<DataLoaderData> NcclTrainer::BuildReplicaBatches(DataLoaderData& batch) const
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
        replicaBatch.input = batch.input
            .slice(0, static_cast<int64_t>(offset), static_cast<int64_t>(end));

        replicaBatch.target = batch.target
            .slice(0, static_cast<int64_t>(offset), static_cast<int64_t>(end));

        for (const auto& [name, value] : batch.additionalData)
        {
            TORCH_CHECK(
                value.dim() > 0 && static_cast<size_t>(value.size(0)) == totalBatchSize,
                "Additional tensor '", name, "' has no matching batch dimension");
            
            replicaBatch.additionalData.emplace(
                name,
                value.slice(0, static_cast<int64_t>(offset), static_cast<int64_t>(end))
            );
        }

        batches.push_back(std::move(replicaBatch));
        offset = end;
    }
    return batches;
}

void NcclTrainer::RunTrainStepsFull(std::vector<torch::Tensor>& losses, bool canUpdate)
{
    TORCH_CHECK(losses.size() == replicaModels.size(), "Expected one loss per replica");
    RunOnReplicas([&](size_t device)
    {
#ifdef _DEBUG
        CheckLoss(losses[device], replicaModels[device]);
#endif
        losses[device].backward();
    });

    if (canUpdate)
    {
        RunOptimizerFull();
    }
}

void NcclTrainer::RunOptimizerFull()
{
    nccl->AllReduceGradients(ParametersByDevice());
    

    RunOnReplicas([&](size_t device)
    {
        auto& model = replicaModels[device];

        auto& optimizer = model->optimizer;
        
        if (sets.clippingFn)
        {
            sets.clippingFn(model->parameters());
        }
        optimizer->step();
        if (model->scheduler)
        {
            model->scheduler->Step();
        }

        optimizer->zero_grad();
    });
}

void NcclTrainer::RunTrainStepsAutocast(std::vector<torch::Tensor>& losses, bool canUpdate)
{
    TORCH_CHECK(losses.size() == replicaModels.size(), "Expected one loss per replica");
    RunOnReplicas([&](size_t device)
    {
#ifdef _DEBUG
        CheckLoss(losses[device], replicaModels[device]);
#endif
        scalers[device]->scale(losses[device]).backward();
    });

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

    RunOnReplicas([&](size_t device)
    {        
        scalers[device]->unscale_(*replicaModels[device]->optimizer);
    });

    if (!globalNonFinite)
    {
        nccl->AllReduceGradients(ParametersByDevice());
    }
    
    RunOnReplicas([&](size_t device)
    {
        auto& model = replicaModels[device];

        auto& optimizer = model->optimizer;
        
        if (sets.clippingFn && !globalNonFinite)
        {            
            sets.clippingFn(model->parameters());
        }

        scalers[device]->step(*optimizer);
        scalers[device]->update();

        if (model->scheduler)
        {
            model->scheduler->Step();
        }

        optimizer->zero_grad();
    });
}

void NcclTrainer::RunStep(DataLoaderData& batch, bool canUpdate)
{
    auto replicaBatches = BuildReplicaBatches(batch);
    const size_t totalBatchSize = batch.GetBatchSize();
    TORCH_CHECK(totalBatchSize > 0, "Cannot train on an empty dataloader batch");
    std::vector<torch::Tensor> losses(replicaModels.size());
    std::vector<torch::Tensor> metricLosses(replicaModels.size());
    std::vector<torch::Tensor> predictions(replicaModels.size());

    RunOnReplicas([&](size_t device)
    {
        auto& replicaBatch = replicaBatches[device];
        torch::Tensor loss;
        if (replicaBatch.GetBatchSize() == 0)
        {
            // Keep every communicator participating when a short final batch
            // has fewer samples than GPUs. This rank contributes zero.

            for (const auto& parameter : replicaModels[device]->parameters())
            {
                if (parameter.requires_grad() && (parameter.numel() > 0))
                {
                    // An empty sum stays zero even if the parameter has NaNs.
                    loss = parameter.reshape({ -1 }).slice(0, 0, 0).sum();
                    break;
                }
            }
            TORCH_CHECK(loss.defined(), "Model replica ", device, " has no trainable parameters");
        }
        else
        {
            const torch::Device targetDevice(torch::kCUDA, static_cast<c10::DeviceIndex>(device));
            auto transfer = [&](torch::Tensor& tensor)
            {
                tensor = tensor.to(targetDevice, tensor.dtype(), sets.perf.useNonBlockingTransfers);
            };
            transfer(replicaBatch.input);
            transfer(replicaBatch.target);
            for (auto& entry : replicaBatch.additionalData)
            {
                transfer(entry.second);
            }

            // NCCL averages replica gradients. Weight uneven shards so that the
            // result still equals the gradient of the complete input batch.
            const double shardWeight = replicaModels.size() == 1
                ? 1.0
                : static_cast<double>(replicaModels.size()) *
                static_cast<double>(replicaBatches[device].GetBatchSize()) /
                static_cast<double>(totalBatchSize);

            loss = Runner::ForwardAndLoss(replicaBatch, replicaModels[device],
                metrics ? &predictions[device] : nullptr);
            TORCH_CHECK(loss.defined(), "NCCL training requires a loss function");
            metricLosses[device] = loss.detach();
            loss = loss * shardWeight;
        }

#ifdef _DEBUG
        CheckLoss(loss, replicaModels[device]);
#endif
        if (sets.perf.enableAutoCast)
        {
            scalers[device]->scale(loss).backward();
        }
        else
        {
            loss.backward();
        }
        losses[device] = loss.detach();
    });

    if (canUpdate)
    {
        if (sets.perf.enableAutoCast)
        {
            RunOptimizerAutoCast();
        }
        else
        {
            RunOptimizerFull();
        }
    }

    // Shared metrics are updated only by the coordinator, after every replica
    // has submitted backward and optimizer work. Retain no autograd graphs.
    for (size_t device = 0; device < replicaModels.size(); ++device)
    {
        if (metricLosses[device].defined())
        {
            const c10::cuda::CUDAGuard deviceGuard(static_cast<c10::DeviceIndex>(device));
            UpdateMetrics(replicaBatches[device], metricLosses[device], predictions[device]);
        }
    }

    // Sampling the displayed loss avoids forcing the CPU to wait every batch.
    constexpr size_t lossLogInterval = 10;
    if (batchIndex % lossLogInterval == 0 || batchIndex + 1 == dataLoaderBatchesCount)
    {
        lastProgressLoss = 0.0f;
        for (const auto& loss : losses)
        {
            lastProgressLoss += loss.item<float>() / static_cast<float>(losses.size());
        }
    }
    ProgressLoss(lastProgressLoss);
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
        const c10::cuda::CUDAGuard deviceGuard(0);
        Runner::PrepareBatch(batch);
    }
    // Multi-GPU batches remain in pinned host memory and are split before
    // being copied directly to their destination devices.
}

void NcclTrainer::PrepareModel()
{
    RunOnReplicas([&](size_t device)
    {
        replicaModels[device]->to(
            torch::Device(torch::kCUDA, static_cast<int8_t>(device)));
    });
}

void NcclTrainer::OnEpochStart()
{
    const c10::cuda::CUDAGuard deviceGuard(0);
    Runner::OnEpochStart();
    lastProgressLoss = 0.0f;

    RunOnReplicas([&](size_t device)
    {
        replicaModels[device]->train();
    });

    torch::autograd::GradMode::set_enabled(true);
}

void NcclTrainer::OnModelEpochStart()
{
    RunOnReplicas([&](size_t device)
    {
        replicaModels[device]->OnEpochStart();
    });

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

void NcclTrainer::OnModelBatchStart()
{
    RunOnReplicas([&](size_t device) { replicaModels[device]->OnBatchStart(); });
}

void NcclTrainer::OnModelBatchEnd()
{
    RunOnReplicas([&](size_t device) { replicaModels[device]->OnBatchEnd(); });
}

void NcclTrainer::OnModelEpochEnd()
{
    RunOnReplicas([&](size_t device)
    {
        replicaModels[device]->OnEpochEnd();
        // Epoch boundaries may hand the primary model to validation or saving.
        c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device)).synchronize();
    });
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

    this->RunStep(batch, canUpdate);
    
}

void NcclTrainer::OnEpochEnd()
{
    const c10::cuda::CUDAGuard deviceGuard(0);
    Runner::OnEpochEnd();

    if ((this->metrics) && (this->metrics->IsBetterThan(this->bestMetrics)))
    {
        SnapshotSaver saver(this->model.get());
        saver.Save(sets.pretrainedManager);
        this->bestMetrics = this->metrics;
    }
}

#endif
