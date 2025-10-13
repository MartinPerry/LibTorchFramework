#ifndef METRICS_DEFAULT_H
#define METRICS_DEFAULT_H

#include <vector>
#include <unordered_map>

#include <torch/torch.h>

class MetricsDefault
{
public:
	static inline size_t PROCESS_EVERY_NTH_INPUT = 1;

	MetricsDefault();
	virtual ~MetricsDefault();

	virtual void Reset();

	//todo
	//void SetPredictionEvaluator();

	virtual bool IsBetterThan(std::shared_ptr<MetricsDefault> other) const;

	virtual std::unordered_map<std::string, float> GetResultExtended() const;
	virtual void Save(const std::string& filePath) const;

	void UpdateProcessCounter();
	bool CanProcess() const;

	void AddLoss(at::Tensor loss);
	void AddDataIndices(const std::vector<int64_t>& indices);
	void AddPredictionTarget(at::Tensor pred, at::Tensor target, bool firstDimensionIsBatch = true);

	float GetMeanLoss() const;

protected:
	size_t batchesCount;
	size_t processCounter;

	std::vector<float> losses;
	mutable std::optional<float> meanLoss;

	std::vector<int64_t> dataIndices;

	at::Tensor pred; 
	at::Tensor target;

	virtual void Evaluate();
};

#endif