#ifndef CUDA_GRAPH_HELPER_H
#define CUDA_GRAPH_HELPER_H

class Trainer;

#include <memory>
#include <optional>

#include <torch/torch.h>
#include <ATen/cuda/CUDAGraph.h>

#include "../InputProcessing/DataLoaderData.h"

class CudaGraphHelper
{
public:
    struct CudaGraphTrainState
    {                
        int warmupStepsRemaining = 0;
        bool captureOptimizerStep = false;
        bool allowDynamicScalerGraph = false;
        bool warnedDynamicScalerGraph = false;

        bool captured = false;
        bool warnedUnsupportedBatch = false;
        bool warnedCaptureFailure = false;
        bool warnedWarmup = false;
        std::optional<DataLoaderData> staticBatch = std::nullopt;
        at::Tensor staticLoss;
        std::unique_ptr<at::cuda::CUDAGraph> graph;
    };

    CudaGraphHelper(Trainer* trainer, int warmupSteps, 
        bool captureOptimizerStep, bool allowDynamicScalerGraph);
    ~CudaGraphHelper() = default;

    CudaGraphTrainState& GetCudaGraphState();

    void Run(DataLoaderData& batch, std::shared_ptr<torch::optim::Optimizer> optimizer);
    void RunCapture(DataLoaderData& batch, std::shared_ptr<torch::optim::Optimizer> optimizer);
    void RunReplay(std::shared_ptr<torch::optim::Optimizer> optimizer);

    
protected:
    CudaGraphTrainState state;

    Trainer* trainer;

    static bool IsSameTensorLayout(const torch::Tensor& src, const torch::Tensor& dst);
    static bool InitStaticTensor(const torch::Tensor& src, torch::Tensor& dst);
    static bool CopyToStaticTensor(const torch::Tensor& src, torch::Tensor& dst);
    static bool InitStaticBatch(DataLoaderData& src, DataLoaderData& dst);
    static bool CopyBatchToStatic(DataLoaderData& src, DataLoaderData& dst);

    bool CheckNeedCapture(DataLoaderData& batch);
};


#endif
