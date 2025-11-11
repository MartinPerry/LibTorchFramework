#ifndef MODULES_OPTIONS_H
#define MODULES_OPTIONS_H

#include <torch/torch.h>

struct ResampleOptions 
{
	ResampleOptions(int64_t inChannels, int64_t outChannels, int64_t scaleFactor) : 
		inChannels_(inChannels),
		outChannels_(outChannels),
		scaleFactor_(scaleFactor)
	{}

	TORCH_ARG(int64_t, inChannels);

	TORCH_ARG(int64_t, outChannels);

	TORCH_ARG(int64_t, scaleFactor);

	TORCH_ARG(int64_t, kernelSize) = 1;
	TORCH_ARG(int64_t, padding) = 0;
	TORCH_ARG(int64_t, dilation) = 1;
};

//=============================================================

struct ResidualBlockOptions
{
	ResidualBlockOptions(int64_t inChannels, int64_t outChannels) :
		inChannels_(inChannels),
		outChannels_(outChannels)
	{
	}

	TORCH_ARG(int64_t, inChannels);

	TORCH_ARG(int64_t, outChannels);


	TORCH_ARG(int64_t, stride) = 1;
	TORCH_ARG(int64_t, dilation) = 1;
	TORCH_ARG(int64_t, outExpansion) = 0;
};

#endif
