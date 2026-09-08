#ifndef ABSTRACT_MODEL_H
#define ABSTRACT_MODEL_H

class FreezeInfo;
struct DataLoaderData;
class Trainer;
class NcclTrainer;

#include <vector>

#include <torch/torch.h>

#include <Utils/Logger.h>

class AbstractModel : public torch::nn::Module
{
public:

	AbstractModel();
	virtual ~AbstractModel();

	virtual const char* GetName() const = 0;
		
	template <typename OptimType, typename Options>
	void CreateOptimizer(const Options& options = {}, bool onlyGradientParams = true);

	template <typename SchedulerType, typename... Params>
	void CreateScheduler(const Params&... p);
	
	void RemoveOptimizer();
	void RemoveScheduler();

	void SetFrozen(std::shared_ptr<FreezeInfo> freezeInfo) const;

	virtual std::vector<torch::Tensor> RunForward(DataLoaderData& batch) = 0;

	virtual void OnBatchStart();
	virtual void OnBatchEnd();

	virtual void OnEpochStart();
	virtual void OnEpochEnd();

	friend class Trainer;
	friend class NcclTrainer;

protected:

	std::shared_ptr<torch::optim::Optimizer> optimizer;
	std::shared_ptr<torch::optim::LRScheduler> scheduler;
};

template <typename OptimType, typename Options>
void AbstractModel::CreateOptimizer(const Options& options, bool onlyGradientParams)
{	
	
	if (onlyGradientParams)
	{
		torch::autograd::variable_list paramGroups;		
		for (auto& p : this->parameters())
		{
			if (p.requires_grad())
			{
				paramGroups.emplace_back(p);				
			}
		}

		this->optimizer = std::make_shared<OptimType>(paramGroups, options);
	}
	else
	{
		this->optimizer = std::make_shared<OptimType>(this->parameters(), options);
	}	
	
}

template <typename SchedulerType, typename... Params>
void AbstractModel::CreateScheduler(const Params&... p)
{
	if (this->optimizer == nullptr)
	{
		MY_LOG_ERROR("Scheduler can be created only after optimizer is inited");
		return;
	}

	this->scheduler = std::make_shared<SchedulerType>(this->optimizer, p...);
}

#endif