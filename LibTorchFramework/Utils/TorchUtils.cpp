#include "./TorchUtils.h"

#include <Utils/Logger.h>

bool TorchUtils::IsBf16Supported()
{
	if (!torch::cuda::is_available())
	{
		return false;
	}

	//return torch::cuda::is_bf16_supported();
	

	// Fallback: check device capability (Ampere+ => cc >= 8.0)
	///const auto props = torch::cuda::
	//return props->major >= 8;
	return false;
}

/// <summary>
/// Print tensor info
/// </summary>
/// <param name="desc"></param>
/// <param name="t"></param>
void TorchUtils::TensorPrintInfo(const std::string& name, const torch::Tensor& x)
{
	auto tmp = x.sizes();

	std::string shapeInfo = "[";
	for (size_t i = 0; i < tmp.size(); i++)
	{
		shapeInfo += std::to_string(tmp[i]);
		shapeInfo += ", ";
	}

	if (shapeInfo.size() > 2)
	{
		shapeInfo.pop_back();
		shapeInfo.pop_back();
	}
	shapeInfo += "]";

	std::string gradName = "";
	if (const auto& g = x.grad_fn())
	{
		//crash on gName when freeing		
		gradName = g->name();				
	}

	MY_LOG_INFO(
		"%s | requires_grad=%d | grad_fn=%s | dtype=%s | shape=%s",
		name.c_str(),
		x.requires_grad(),
		x.grad_fn() ? gradName.c_str() : "-",
		std::string(x.dtype().name()).c_str(),
		shapeInfo.c_str()
	);
}

