#ifndef TORCH_IMAGE_UTILS_H
#define TORCH_IMAGE_UTILS_H

#include <optional>
#include <vector>

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

	template <typename T>
	struct MappingRange
	{
		T dataMin = T{ 0 };
		T dataMax = std::is_same_v<T, uint8_t> ? T{ 255 } : T{ 1 };
		float minMapTo = 0.0f;
		float maxMapTo = 1.0f;
	};

	struct IntervalMapping
	{
		bool enabled = true;
		std::optional<MappingRange<float>> mapRange = std::nullopt; //if nullopt, auto-calculated		
	};

	struct TensorsToImageSettings
	{
		SequenceFormat seqFormat = SequenceFormat::B_S;
		int chanCount = -1;
		int w = -1;
		int h = -1;
		int borderSize = 0;
		uint8_t backgroundValue = 255;
		IntervalMapping intervalMapping = IntervalMapping{};
		std::optional<std::string> colorMappingFileName = std::nullopt;
	};

	static const TensorsToImageSettings DEFAULT_TENSOR_TO_IMAGE;

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
		const MappingRange<uint8_t>& range = {});

	template <typename T>
	static TENSOR_VEC_RET_VAL(T) LoadImageAs(
		Image2d<float>& img,
		int chanCount,
		int w,
		int h);

	template <typename T>
	static std::vector<float> ImageToVector_CHW(
		const Image2d<T>& v,
		const MappingRange<T>& range);

	static Image2d<uint8_t> TensorToImage(at::Tensor t,
		int chanCount = -1,
		int w = -1,
		int h = -1,
		IntervalMapping intervalMapping = IntervalMapping());

	static Image2d<uint8_t> TensorsToImage(at::Tensor t, 
		const TensorsToImageSettings& sets = DEFAULT_TENSOR_TO_IMAGE);

	static std::vector<Image2d<uint8_t>> TensorsToImages(at::Tensor t,
		const TensorsToImageSettings& sets = DEFAULT_TENSOR_TO_IMAGE);

	static Image2d<uint8_t> TensorsToImage(const std::vector<std::vector<torch::Tensor>>& t,
		const TensorsToImageSettings& sets = DEFAULT_TENSOR_TO_IMAGE);

	static std::vector<Image2d<uint8_t>> TensorsToImages(const std::vector<std::vector<torch::Tensor>>& t,
		const TensorsToImageSettings& sets = DEFAULT_TENSOR_TO_IMAGE);

	static std::vector<std::vector<torch::Tensor>> MergeTensorsToRows(
		const std::vector<torch::Tensor>& tensors,
		int maxRowsCount = 4);
};


#endif