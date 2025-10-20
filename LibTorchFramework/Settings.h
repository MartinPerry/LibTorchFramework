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
	using GradientClippingCallback = std::function<double(const torch::autograd::variable_list& params)>;

	torch::DeviceType device = torch::kCPU;

	int epochCount = 10;

	size_t batchSize = 3;

	size_t numWorkers = 0;
	
	LossFnCallback lossFn = nullptr;

	MetricsInitCallback metricsInitFn = nullptr;

	std::shared_ptr<PretrainedManager> pretrainedManager = nullptr;

	PerformanceSettings perf;

	//Gradient Norm Clipping - torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 2.0, norm_type = 2)
	//Gradient Value Clipping - torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value = 1.0)
	GradientClippingCallback clippingFn = nullptr;

	std::optional<int> gradientAccumulationCount = std::nullopt;

	static void PrintCudaInfo()
	{
		MY_LOG_INFO("CUDA device count: %d", torch::cuda::device_count());
		MY_LOG_INFO("CUDA is available: %s", (torch::cuda::is_available() ? "Yes" : "No"));
		MY_LOG_INFO("cuDNN is available: %s", (torch::cuda::cudnn_is_available() ? "Yes" : "No"));
	}
};

#endif
