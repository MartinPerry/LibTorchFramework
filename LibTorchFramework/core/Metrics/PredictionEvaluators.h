#ifndef PREDICTION_EVALUATORS_H
#define PREDICTION_EVALUATORS_H

//Classes used to convert output of model ro real data
//output of model can be sigmoids or other type of data representation
//this way we convert output to real target data

#include <torch/torch.h>

class PredictionEvaluator
{
public:
	PredictionEvaluator() = default;
	virtual ~PredictionEvaluator() = default;

	virtual torch::Tensor Convert(torch::Tensor input) const = 0;
};


class PredictionEvaluatorSigmoid : public PredictionEvaluator
{
public:
	torch::Tensor Convert(torch::Tensor input) const override
	{
		return torch::sigmoid(input);
	}
};



class PredictionEvaluatorMax : public PredictionEvaluator
{
public:
	PredictionEvaluatorMax(int64_t dim) :
		dim(dim)
	{}

	torch::Tensor Convert(torch::Tensor input) const override
	{		
		auto result = torch::max(input, dim);
		return std::get<1>(result);
	}

private:
	int64_t dim;
};

#endif
