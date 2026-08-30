#ifndef NCCL_TRAINER_H
#define NCCL_TRAINER_H

//#define LIBTORCH_FRAMEWORK_HAS_NCCL

#ifdef LIBTORCH_FRAMEWORK_HAS_NCCL

struct Settings;
class MetricsDefault;
class AbstractModel;
struct DataLoaderData;

class CudaGraphHelper;
class NcclTrainerContext;

namespace torch {
	namespace amp {
		class GradScaler;
	}
}

#include <torch/torch.h>

#include "../Runner.h"

class NcclTrainer : public Runner
{
public:	
	
	NcclTrainer(const Settings& sets, std::vector<std::shared_ptr<AbstractModel>> models);
	virtual ~NcclTrainer();
	
	friend class CudaGraphHelper;

protected:

	std::shared_ptr<CudaGraphHelper> cudaGraph;

	std::vector<std::shared_ptr<torch::amp::GradScaler>> scalers;

	std::vector<std::shared_ptr<AbstractModel>> replicaModels;

	std::shared_ptr<MetricsDefault> bestMetrics;

	std::shared_ptr<NcclTrainerContext> nccl;
	bool distributedParametersSynchronized;

	void SelectCudaDevice(size_t device);

	void CheckLoss(at::Tensor loss, const std::shared_ptr<AbstractModel>& activeModel);

	std::vector<std::vector<torch::Tensor>> ParametersByDevice() const;
	std::vector<DataLoaderData> BuildReplicaBatches(DataLoaderData& batch) const;

	void RunTrainStepsFull(std::vector<torch::Tensor>& losses, bool canUpdate);
	void RunTrainStepsAutocast(std::vector<torch::Tensor>& losses, bool canUpdate);

	void RunOptimizerFull();
	void RunOptimizerAutoCast();

	void RunStep(DataLoaderData& batch, bool canUpdate);

	torch::Tensor ForwardAndLoss(DataLoaderData& batch, std::shared_ptr<AbstractModel> model);

	void ProgressLoss(float loss);

	virtual void PrepareBatch(DataLoaderData& batch) override;
	virtual void PrepareModel() override;
	virtual void OnModelEpochStart() override;
	virtual void OnEpochStart() override;
	virtual void ProcessBatch(DataLoaderData& batch) override;
	virtual void OnEpochEnd() override;
};

#endif

#endif
