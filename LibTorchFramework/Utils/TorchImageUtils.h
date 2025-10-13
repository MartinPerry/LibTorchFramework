#ifndef TORCH_IMAGE_UTILS_H
#define TORCH_IMAGE_UTILS_H

#include <torch/torch.h>

#include <RasterData/Image2d.h>

struct ImageSize
{
	uint16_t channels;
	uint16_t width;
	uint16_t height;

	// Constructor
	ImageSize(uint16_t c, uint16_t w, uint16_t h)
		: channels(c), width(w), height(h) {
	}

	// Optional: conversion to std::vector<uint16_t>
	operator std::vector<uint16_t>() const {
		return { channels, width, height };
	}
};

class TorchImageUtils
{
public:
	enum class SequenceFormat 
	{ 
		B_S = 0, 
		S_B = 1 
	};

	static Image2d<uint8_t> TensorToImage(at::Tensor t,
		int chanCount = -1,
		int w = -1,
		int h = -1,
		bool intervalMapping = true);

	static Image2d<uint8_t> TensorsToImage(at::Tensor t,
		SequenceFormat seqFormat = SequenceFormat::B_S,
		int chanCount = -1,
		int w = -1,
		int h = -1,
		int borderSize = 0,
		uint8_t backgroundValue = 255,
		bool intervalMapping = true);

	static Image2d<uint8_t> TensorsToImage(const std::vector<std::vector<torch::Tensor>>& t,
		SequenceFormat seqFormat = SequenceFormat::B_S,
		int chanCount = -1,
		int w = -1,
		int h = -1,
		int borderSize = 0,
		uint8_t backgroundValue = 255,
		bool intervalMapping = true);

	static std::vector<std::vector<torch::Tensor>> MergeTensorsToRows(
		const std::vector<torch::Tensor>& tensors,
		int maxRowsCount = 4);
};

#endif