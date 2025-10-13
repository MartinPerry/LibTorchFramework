#ifndef METRICS_IMAGE_H
#define METRICS_IMAGE_H

#include <list>
#include <tuple>
#include <unordered_map>

#include <torch/torch.h>

#include "./MetricsDefault.h"

class MetricsImage : public MetricsDefault
{
public:
	enum class MetricsType 
	{		
		SEGMENTATION,
		UNKNOWN
	};

	MetricsImage();
	MetricsImage(MetricsType mType);
	~MetricsImage() = default;

	std::unordered_map<std::string, float> GetResultExtended() const override;
	void Save(const std::string& filePath) const override;

	void Reset() override;

protected:
	MetricsType mType;

	int keepImages;
	std::list<std::tuple<at::Tensor, at::Tensor>> images;

	float iPosAll; //intersection all
	float uPosAll; //union all
	float iInvAll; //intersection inverse all
	float uInvAll; //union inverse all

	float runningMae;
	float runningMse;

	size_t pixelsCount;

	float threshold;

	std::string BuildPath(const std::string& path,
		int fileIndex,
		const std::string& extension,
		bool extensionSeparateDir = false) const;

	void AddImages(at::Tensor p, at::Tensor t);
	void RunningRmseMae(at::Tensor p, at::Tensor t);
	void JaccardIndexBinary(at::Tensor p, at::Tensor t, bool mergeBatches = true);

	std::tuple<float, float, float, float> CalcIntersectUnions(torch::Tensor p, torch::Tensor t, bool mergeBatches) const;
	std::pair<torch::Tensor, torch::Tensor> IouInverse(const torch::Tensor& p, const torch::Tensor& t) const;
	std::pair<torch::Tensor, torch::Tensor> Iou(const torch::Tensor& p, const torch::Tensor& t) const;

	virtual void Evaluate() override;
};

#endif
