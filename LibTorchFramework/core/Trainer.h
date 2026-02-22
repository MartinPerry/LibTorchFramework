#ifndef TRAINER_H
#define TRAINER_H

struct Settings;
class MetricsDefault;
class AbstractModel;
struct DataLoaderData;

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
	

protected:

	std::shared_ptr<torch::amp::GradScaler> scaler;

	std::shared_ptr<MetricsDefault> bestMetrics;

	void CheckLoss(at::Tensor loss);

	void RunTrainStepsFull(at::Tensor loss, std::shared_ptr<torch::optim::Optimizer> optimizer);
	void RunTrainStepsAutocast(at::Tensor loss, std::shared_ptr<torch::optim::Optimizer> optimizer);

	virtual void OnEpochStart() override;
	virtual void ProcessBatch(DataLoaderData& batch) override;
	virtual void OnEpochEnd() override;
};



#endif