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
void TorchUtils::TensorPrintInfo(const char* desc, const at::Tensor& t)
{
	auto tmp = t.sizes();

	std::string res = "{";
	for (int i = 0; i < tmp.size(); i++)
	{
		res += std::to_string(tmp[i]);
		res += ", ";
	}

	if (res.size() > 2)
	{
		res.pop_back();
		res.pop_back();
	}
	res += "}";

	std::string info = desc;
	info += ": ";
	info += res.c_str();
	info += ", ";

	if (const auto& g = t.grad_fn())
	{
		//crash on gName when freeing
		info += "grad name: ";
		info += g->name();
		info += ", ";
		//std::cout << "Grad: " << gName << std::endl;
	}
	else if (t.requires_grad())
	{
		info += "requires_grad";
		info += ", ";
	}

	MY_LOG_INFO("%s", info.c_str());
}

