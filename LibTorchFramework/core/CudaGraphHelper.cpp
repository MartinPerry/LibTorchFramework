#include "./CudaGraphHelper.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <string>


#include "./Modules/gradscaler.hpp"
#include "./Trainer.h"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

CudaGraphHelper::CudaGraphHelper(Trainer* trainer, int warmupSteps,
    bool captureOptimizerStep, bool allowDynamicScalerGraph) : 
    trainer(trainer)
{    
    state.warmupStepsRemaining = warmupSteps;
    state.captureOptimizerStep = captureOptimizerStep;
    state.allowDynamicScalerGraph = allowDynamicScalerGraph;

}

CudaGraphHelper::CudaGraphTrainState& CudaGraphHelper::GetCudaGraphState()
{
	return this->state;
}



bool CudaGraphHelper::IsSameTensorLayout(const torch::Tensor& src, const torch::Tensor& dst)
{
	return src.defined() &&
		dst.defined() &&
		(src.device() == dst.device()) &&
		(src.scalar_type() == dst.scalar_type()) &&
		(src.sizes() == dst.sizes());
}

bool CudaGraphHelper::InitStaticTensor(const torch::Tensor& src, torch::Tensor& dst)
{
	if ((!src.defined()) || (!src.is_cuda()))
	{
		return false;
	}

	dst = torch::empty_like(src);
	dst.copy_(src, /*non_blocking=*/true);
	return true;
}

bool CudaGraphHelper::CopyToStaticTensor(const torch::Tensor& src, torch::Tensor& dst)
{
	if (!CudaGraphHelper::IsSameTensorLayout(src, dst))
	{
		return false;
	}

	dst.copy_(src, /*non_blocking=*/true);
	return true;
}

bool CudaGraphHelper::InitStaticBatch(DataLoaderData& src, DataLoaderData& dst)
{
	dst = src;

	if (CudaGraphHelper::InitStaticTensor(src.input, dst.input) == false)
	{
		return false;
	}

	if (src.target.defined())
	{
		if (CudaGraphHelper::InitStaticTensor(src.target, dst.target) == false)
		{
			return false;
		}
	}
	else
	{
		dst.target = torch::Tensor();
	}
		
	return true;
}

/// <summary>
/// CopyBatchToStatic copies the current batch tensors into the preallocated �static� batch buffers used by CUDA Graph replay.
/// It is used to:
/// reuse fixed CUDA memory addresses between steps(required by CUDA Graphs),
/// check that shape / dtype / device still match the captured graph,
/// trigger recapture if layout changed.
/// </summary>
/// <param name="src"></param>
/// <param name="dst"></param>
/// <returns></returns>
bool CudaGraphHelper::CopyBatchToStatic(DataLoaderData& src, DataLoaderData& dst)
{	
	if (CudaGraphHelper::CopyToStaticTensor(src.input, dst.input) == false)
	{
		return false;
	}
	
	
	if (src.target.defined() != dst.target.defined())
	{
		return false;
	}
	if (src.target.defined() && 
		(CudaGraphHelper::CopyToStaticTensor(src.target, dst.target) == false))
	{
		return false;
	}
	
	return true;
}


void CudaGraphHelper::Run(DataLoaderData& batch, std::shared_ptr<torch::optim::Optimizer> optimizer)
{
    if (optimizer != nullptr)
    {
        if ((trainer->sets.perf.enableAutoCast) &&
            (state.captureOptimizerStep == false) &&
            (state.allowDynamicScalerGraph == false))
        {
            if (state.warnedDynamicScalerGraph == false)
            {
                MY_LOG_WARNING(
                    "CUDA Graphs are disabled for this run: autocast + non-captured optimizer uses dynamic GradScaler state. "
                    "Enable LLAMA_CUDA_GRAPHS_ALLOW_DYNAMIC_SCALER=1 to force this experimental path."
                );
                state.warnedDynamicScalerGraph = true;
            }
            
            trainer->RunStep(batch, optimizer);
            return;
        }
        
        bool looksSupported = batch.input.defined() && batch.input.is_cuda();
        if (looksSupported == false)
        {
            if (state.warnedUnsupportedBatch == false)
            {
                MY_LOG_WARNING("LLAMA_USE_CUDA_GRAPHS=1 was requested, but batch layout/device is unsupported. "
                    "Falling back to classic training.");
                state.warnedUnsupportedBatch = true;
            }
        }
        else
        {
            if (this->CheckNeedCapture(batch))
            {
                this->RunCapture(batch, optimizer);
            }
            else
            {
                this->RunReplay(optimizer);
                return;
            }
        }
    }

    trainer->RunStep(batch, optimizer);
}

bool CudaGraphHelper::CheckNeedCapture(DataLoaderData& batch)
{
    bool needsCapture = (state.captured == false) || (!state.staticBatch.has_value());
    if ((needsCapture == false) && (CudaGraphHelper::CopyBatchToStatic(batch, *state.staticBatch) == false))
    {
        needsCapture = true;
    }

    return needsCapture;
}

void CudaGraphHelper::RunCapture(DataLoaderData& batch,
    std::shared_ptr<torch::optim::Optimizer> optimizer)
{
    bool needsWarmup = (state.warmupStepsRemaining > 0) || state.captured;
    if (needsWarmup)
    {
        if ((state.warnedWarmup == false) && (state.warmupStepsRemaining > 0))
        {
            MY_LOG_INFO("CUDA Graph warmup is active (%d eager optimizer steps remaining before capture).", 
                state.warmupStepsRemaining);
            state.warnedWarmup = true;
        }

        if (state.warmupStepsRemaining > 0)
        {
            --state.warmupStepsRemaining;
        }

        state.captured = false;
        state.graph.reset();
        state.staticBatch.reset();
        
        trainer->RunStep(batch, optimizer);

        return;
    }

    state.captured = false;
    state.staticBatch = batch;
    const auto captureStream = at::cuda::getStreamFromPool(
        /*isHighPriority=*/false,
        batch.input.device().index());
    c10::cuda::CUDAStreamGuard captureStreamGuard(captureStream);
    if (CudaGraphHelper::InitStaticBatch(batch, *state.staticBatch))
    {
        try
        {            
            state.graph = std::make_unique<at::cuda::CUDAGraph>();            
            state.graph->capture_begin();

            state.staticLoss = trainer->ForwardAndLoss(*state.staticBatch);
            if (state.captureOptimizerStep)
            {
                if (trainer->sets.perf.enableAutoCast)
                {
                    trainer->RunTrainStepsAutocast(state.staticLoss, optimizer);
                }
                else
                {
                    trainer->RunTrainStepsFull(state.staticLoss, optimizer);
                }
            }
            else
            {
                if (trainer->sets.perf.enableAutoCast)
                {                    
                    auto scaledLoss = trainer->scaler->scale(state.staticLoss);
                    scaledLoss.backward();
                }
                else
                {
                    state.staticLoss.backward();
                }
            }

            state.graph->capture_end();
            state.captured = true;
            if (state.captureOptimizerStep == false)
            {
                if (trainer->sets.perf.enableAutoCast)
                {
                    trainer->RunOptimizerAutoCast(optimizer);
                }
                else 
                {
                    trainer->RunOptimizerFull(optimizer);
                }                
            }            
            trainer->ProgressLoss(state.staticLoss.item().toFloat());
            return;
        }
        catch (const std::exception& ex)
        {
            state.graph.reset();
            state.staticBatch.reset();

            if (state.warnedCaptureFailure == false)
            {
                MY_LOG_WARNING("CUDA Graph capture failed (%s). CUDA Graphs are disabled for this run; "
                    "falling back to eager training.", ex.what());
                state.warnedCaptureFailure = true;
            }
        }
    }
    else if (state.warnedUnsupportedBatch == false)
    {
        MY_LOG_WARNING("LLAMA_USE_CUDA_GRAPHS=1 was requested, but static CUDA batch buffers could not be created."
            "Falling back to eager training.");
        state.warnedUnsupportedBatch = true;
    }
}

void CudaGraphHelper::RunReplay(std::shared_ptr<torch::optim::Optimizer> optimizer)
{
    state.graph->replay();
    if (state.captureOptimizerStep == false)
    {
        if (trainer->sets.perf.enableAutoCast)
        {
            trainer->RunOptimizerAutoCast(optimizer);
        }
        else
        {
            trainer->RunOptimizerFull(optimizer);
        }
    }

    trainer->ProgressLoss(state.staticLoss.item().toFloat());    
}

