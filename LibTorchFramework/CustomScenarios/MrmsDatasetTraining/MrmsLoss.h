#ifndef MRMS_LOSS_H
#define MRMS_LOSS_H

#include <vector>
#include <optional>

#include <torch/torch.h>

#include "../../core/Modules/LossFunctions/SSIMLoss.h"
#include "../../core/Modules/LossFunctions/FocalFrequencyLoss.h"

namespace CustomScenarios
{
	namespace MrmsTraining
	{
		struct CustomSimVpLossImpl : public torch::nn::Module
		{
			torch::nn::MSELoss mseLoss;
			FocalFrequencyLoss ffLoss;
			SSIMLoss ssimLoss;

			CustomSimVpLossImpl(int chanCount);

			torch::Tensor forward(const torch::Tensor& pred, const torch::Tensor& target);
		};

		TORCH_MODULE(CustomSimVpLoss);
	}
}


#endif