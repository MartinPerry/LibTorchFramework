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

	virtual std::vector<at::Tensor> RunForward(DataLoaderData& batch) = 0;

	virtual void OnBatchStart();
	virtual void OnBatchEnd();

	virtual void OnEpochStart();
	virtual void OnEpochEnd();

	friend class Trainer;

protected:

	std::shared_ptr<torch::optim::Optimizer> optimizer;

};

#endif