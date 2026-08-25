#ifndef METRICS_VIDEO_H
#define METRICS_VIDEO_H

#include <list>
#include <tuple>
#include <unordered_map>
#include <optional>

#include <torch/torch.h>

#include "./MetricsImage.h"

class MetricsVideo : public MetricsImage
{
public:
	

	MetricsVideo();	
	~MetricsVideo() = default;
	
	void Save(const std::string& filePath) const override;



protected:
	void CsiThresholds(torch::Tensor p, torch::Tensor t) override;
};

#endif
