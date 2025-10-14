#ifndef TORCH_SETTINGS_H
#define TORCH_SETTINGS_H

class PretrainedManager;
class MetricsDefault;

#include <vector>
#include <memory>
#include <optional>

#include <torch/torch.h>

#include "./Utils/Logger.h"

#include "./PerformanceSettings.h"

struct Settings
{
	using LossFnCallback = std::function<torch::Tensor(const std::vector<torch::Tensor>& output, torch::Tensor target)>;
	using MetricsInitCallback = std::function<std::shared_ptr<MetricsDefault>()>;

	torch::DeviceType device = torch::kCPU;

	int epochCount = 10;

	size_t batchSize = 3;

	size_t numWorkers = 0;
	
	LossFnCallback lossFn = nullptr;

	MetricsInitCallback metricsInitFn = nullptr;

	std::shared_ptr<PretrainedManager> pretrainedManager = nullptr;

	PerformanceSettings perf;

	std::optional<int> gradientAccumulationCount = std::nullopt;

	static void PrintCudaInfo()
	{
		MY_LOG_INFO("CUDA device count: %d", torch::cuda::device_count());
		MY_LOG_INFO("CUDA is available: %s", (torch::cuda::is_available() ? "Yes" : "No"));
		MY_LOG_INFO("cuDNN is available: %s", (torch::cuda::cudnn_is_available() ? "Yes" : "No"));
	}
};

#endif
