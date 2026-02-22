#ifndef MODULES_OPTIONS_H
#define MODULES_OPTIONS_H

#include "../../Utils/HelperMacros.h"

struct ResampleOptions 
{
	ResampleOptions(int64_t inChannels, int64_t outChannels, int64_t scaleFactor) : 
		inChannels_(inChannels),
		outChannels_(outChannels),
		scaleFactor_(scaleFactor)
	{}

	STRUCT_ARG(int64_t, inChannels);

	STRUCT_ARG(int64_t, outChannels);

	STRUCT_ARG(int64_t, scaleFactor);

	STRUCT_ARG(int64_t, kernelSize) = 1;
	STRUCT_ARG(int64_t, padding) = 0;
	STRUCT_ARG(int64_t, dilation) = 1;
};

//=============================================================

struct ResidualBlockOptions
{
	ResidualBlockOptions(int64_t inChannels, int64_t outChannels) :
		inChannels_(inChannels),
		outChannels_(outChannels)
	{
	}

	STRUCT_ARG(int64_t, inChannels);

	STRUCT_ARG(int64_t, outChannels);


	STRUCT_ARG(int64_t, stride) = 1;
	STRUCT_ARG(int64_t, dilation) = 1;
	STRUCT_ARG(int64_t, outExpansion) = 0;
};

#endif
