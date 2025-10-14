#ifndef ABSTRACT_MODEL_H
#define ABSTRACT_MODEL_H

struct DataLoaderData;
class Trainer;

#include <vector>

#include <torch/torch.h>

class AbstractModel : public torch::nn::Module
{
public:
	AbstractModel();
	virtual ~AbstractModel();

	virtual const char* GetName() const = 0;
	
	template <typename OptimType, typename Options>
	void CreateOptimizer(const Options& options = {});

	void RemoveOptimizer();

	virtual std::vector<torch::Tensor> RunForward(DataLoaderData& batch) = 0;

	virtual void OnBatchStart();
	virtual void OnBatchEnd();

	virtual void OnEpochStart();
	virtual void OnEpochEnd();

	friend class Trainer;

protected:

	std::shared_ptr<torch::optim::Optimizer> optimizer;
};

template <typename OptimType, typename Options>
void AbstractModel::CreateOptimizer(const Options& options)
{
	this->optimizer = std::make_shared<OptimType>(this->parameters(), options);
}

#endif