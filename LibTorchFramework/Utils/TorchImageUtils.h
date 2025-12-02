#ifndef TORCH_IMAGE_UTILS_H
#define TORCH_IMAGE_UTILS_H

#include <optional>

#include <torch/torch.h>

#include <RasterData/Image2d.h>

//=====================================================================

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

//=====================================================================

#define TENSOR_VEC_RET_VAL(T) \
	typename std::enable_if< \
		std::is_same<T, torch::Tensor>::value || \
		std::is_same<T, std::vector<float>>::value, \
	T>::type

class TorchImageUtils
{
public:
	enum class SequenceFormat 
	{ 
		B_S = 0, 
		S_B = 1 
	};

	struct MappingRange 
	{
		uint8_t dataMin = 0;
		uint8_t dataMax = 255;
		float minMapTo = 0.0f;
		float maxMapTo = 1.0f;
	};

	struct TensorsToImageSettings
	{
		SequenceFormat seqFormat = SequenceFormat::B_S;
		int chanCount = -1;
		int w = -1;
		int h = -1;
		int borderSize = 0;
		uint8_t backgroundValue = 255;
		bool intervalMapping = true;
		std::optional<std::string> colorMappingFileName = std::nullopt;
	};

	template <typename T>
	static TENSOR_VEC_RET_VAL(T) LoadImageAs(
		const std::string& imgPath,
		int chanCount,
		int w,
		int h);

	template <typename T>
	static TENSOR_VEC_RET_VAL(T) LoadImageAs(
		Image2d<uint8_t>& img,
		int chanCount,
		int w,
		int h,
		const MappingRange& range = {});

	template <typename T>
	static TENSOR_VEC_RET_VAL(T) LoadImageAs(
		Image2d<float>& img,
		int chanCount,
		int w,
		int h);

	template <typename T>
	static std::vector<float> ImageToVector_CHW(
		const Image2d<T>& v,
		const MappingRange& range);

	static Image2d<uint8_t> TensorToImage(at::Tensor t,
		int chanCount = -1,
		int w = -1,
		int h = -1,
		bool intervalMapping = true);

	static Image2d<uint8_t> TensorsToImage(at::Tensor t, 
		const TensorsToImageSettings& sets = {});

	static Image2d<uint8_t> TensorsToImage(const std::vector<std::vector<torch::Tensor>>& t,
		const TensorsToImageSettings& sets = {});

	static std::vector<std::vector<torch::Tensor>> MergeTensorsToRows(
		const std::vector<torch::Tensor>& tensors,
		int maxRowsCount = 4);
};

#endif