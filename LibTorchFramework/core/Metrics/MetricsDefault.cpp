#include "./MetricsDefault.h"

#include <algorithm>

#include <FileUtils/Writing/TextFileWriter.h>
#include <Utils/cJSON.h>

MetricsDefault::MetricsDefault() : 
	batchesCount(0),
	processCounter(0),
	meanLoss(std::nullopt)
{
}

MetricsDefault::~MetricsDefault()
{
}

void MetricsDefault::Reset()
{
	batchesCount = 0;
	processCounter = 0;

	losses.clear();
	meanLoss = std::nullopt;

	dataIndices.clear();
}

bool MetricsDefault::IsBetterThan(std::shared_ptr<MetricsDefault> other) const
{
	if (other == nullptr)
	{
		return true;
	}

	return this->GetMeanLoss() < other->GetMeanLoss();
}

std::unordered_map<std::string, float> MetricsDefault::GetResultExtended() const
{
	std::unordered_map<std::string, float> res;
	res.try_emplace("loss", this->GetMeanLoss());

	return res;
}

void MetricsDefault::Save(const std::string& filePath) const
{
	std::unordered_map<std::string, float> res = this->GetResultExtended();

	cJSON* root = cJSON_CreateObject();

	for (const auto& [k, v] : res) 
	{
		cJSON_AddNumberToObject(root, k.c_str(), v);
	}
	
	char* json_str = cJSON_Print(root);

	TextFileWriter tf(filePath.c_str());
	tf.Write(json_str);

	tf.Close();

	free(json_str);
	cJSON_Delete(root);
}

/// <summary>
/// Update process counter
/// Call this method after every forward loop
/// </summary>
void MetricsDefault::UpdateProcessCounter()
{
	processCounter += 1;
}

bool MetricsDefault::CanProcess() const
{
	if (processCounter % PROCESS_EVERY_NTH_INPUT != 0)
	{
		return false;
	}

	return true;
}

void MetricsDefault::AddLoss(at::Tensor loss)
{
	losses.push_back(loss.item().toFloat());
	meanLoss = std::nullopt;
}

void MetricsDefault::AddDataIndices(const std::vector<int64_t>& indices)
{
	for (int64_t i : indices)
	{
		dataIndices.push_back(i);
	}
}

void MetricsDefault::AddPredictionTarget(at::Tensor pred, at::Tensor target, bool firstDimensionIsBatch)
{
	this->pred = pred;
	this->target = target;

	if (firstDimensionIsBatch)
	{
		batchesCount += pred.size(0);
	}
	else
	{		
		batchesCount += 1;
	}

	this->Evaluate();
}

float MetricsDefault::GetMeanLoss() const
{
	if (meanLoss.has_value())
	{
		return meanLoss.value();
	}

	meanLoss = std::accumulate(losses.begin(), losses.end(), 0.0f) / static_cast<float>(losses.size());
	return meanLoss.value();
}

/// <summary>
/// Evaluate metrics
/// This method should be implemented in childrens
/// to actually calculate something
/// However, we may require only loss values and nothing more
/// In that case, this can be left empty
/// </summary>
void MetricsDefault::Evaluate()
{
}