#ifndef TRAINER_H
#define TRAINER_H

struct Settings;
class MetricsDefault;
class AbstractModel;
struct DataLoaderData;
class AbstractScheduler;

class CudaGraphHelper;

namespace torch {
	namespace amp {
		class GradScaler;
	}
}

#include <torch/torch.h>

#include "./Runner.h"

class Trainer : public Runner
{
public:	

	Trainer(const Settings& sets, std::shared_ptr<AbstractModel> model);
	virtual ~Trainer();    
	
	friend class CudaGraphHelper;

protected:
	
	struct StepInfo
	{
		at::Tensor loss;
		std::shared_ptr<torch::optim::Optimizer> optimizer;
		std::shared_ptr<AbstractScheduler> scheduler;
	};

	std::shared_ptr<CudaGraphHelper> cudaGraph;

	std::shared_ptr<torch::amp::GradScaler> scaler;

	std::shared_ptr<MetricsDefault> bestMetrics;

	void CheckLoss(at::Tensor loss);

	void RunTrainStepsFull(StepInfo& si);
	void RunTrainStepsAutocast(StepInfo& si);

	void RunOptimizerFull(StepInfo& si);
	void RunOptimizerAutoCast(StepInfo& si);

	void RunStep(DataLoaderData& batch, StepInfo& si);

	void ProgressLoss(float loss);

	virtual void OnEpochStart() override;
	virtual void ProcessBatch(DataLoaderData& batch) override;
	virtual void OnEpochEnd() override;
};



#endif