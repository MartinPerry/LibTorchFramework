#include "./TorchImageUtils.h"

#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <optional>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <memory>

#include <RasterData/ImageResize.h>
#include <RasterData/ImageDrawing.h>
#include <RasterData/Colors/ColorSpace.h>

#include <Utils/Logger.h>

#include "./TorchUtils.h"

/// <summary>
/// Load image from uin8_t format and convert it to 
/// [0, 1] float tensor
/// If error occured, return tensor filled with zeroes
/// </summary>
/// <param name="imgPath"></param>
/// <param name="chanCount"></param>
/// <param name="w"></param>
/// <param name="h"></param>
/// <returns></returns>
template <typename T>
TENSOR_VEC_RET_VAL(T) TorchImageUtils::LoadImageAs(
	const std::string& imgPath,
	int chanCount,
	int w,
	int h)
{
	Image2d<uint8_t> img = Image2d<uint8_t>(imgPath.c_str());
	return TorchImageUtils::LoadImageAs<T>(img, chanCount, w, h);
}

template <typename T>
TENSOR_VEC_RET_VAL(T) TorchImageUtils::LoadImageAs(
	Image2d<uint8_t>& img,
	int chanCount,
	int w,
	int h,
	const MappingRange<uint8_t>& range)
{
	if (img.GetChannelsCount() == 0)
	{
		MY_LOG_ERROR("Failed to load image. Return zero tensor.");

		//failed to load image - return zero tensor
		if constexpr (std::is_same<T, std::vector<float>>::value)
		{
			return std::vector<float>(chanCount * h * w, 0);
		}
		else
		{
			return at::zeros({ chanCount, h, w }, at::kFloat);
		}
	}

	if (chanCount != img.GetChannelsCount())
	{
		if (chanCount == 3)
		{
			auto tmp = ColorSpace::ConvertToRGB(img);
			if (tmp.has_value() == false)
			{
				MY_LOG_ERROR("Failed to convert image. Return zero tensor.");

				//failed to convert image - return zero tensor
				if constexpr (std::is_same<T, std::vector<float>>::value)
				{
					return std::vector<float>(chanCount * h * w, 0);
				}
				else
				{
					return at::zeros({ chanCount, h, w }, at::kFloat);
				}
			}
			img = *tmp;
		}
		else if (chanCount == 1)
		{
			auto tmp = ColorSpace::ConvertToGray(img);
			if (tmp.has_value() == false)
			{
				MY_LOG_ERROR("Failed to convert image. Return zero tensor.");

				//failed to convert image - return zero tensor
				if constexpr (std::is_same<T, std::vector<float>>::value)
				{
					return std::vector<float>(chanCount * h * w, 0);
				}
				else
				{
					return at::zeros({ chanCount, h, w }, at::kFloat);
				}
			}
			img = *tmp;
		}
		else
		{
			MY_LOG_ERROR("Channels count %d not supported", chanCount);
		}
	}

	auto imgFinal = ImageResize<uint8_t>::ResizeBilinear(img, ImageDimension(w ,h));

	if ((imgFinal.GetWidth() != w) && (imgFinal.GetHeight() == h))
	{
		MY_LOG_ERROR("Incorrect image dimension [%d, %d]", imgFinal.GetWidth(), imgFinal.GetHeight());

		//return zero tensor
		if constexpr (std::is_same<T, std::vector<float>>::value)
		{
			return std::vector<float>(chanCount * h * w, 0);
		}
		else
		{
			return at::zeros({ chanCount, h, w }, at::kFloat);
		}
	}

		
	//image at input is stored as RGB/RGB/RGB
	//we need output as RRR/GGG/BBB (CHW)

	std::vector<float> d = TorchImageUtils::ImageToVector_CHW(imgFinal, range);	
	
	//auto imgf = imgFinal.CreateAsMapped<float>(range.dataMin, range.dataMax, range.minMapTo, range.maxMapTo);

	if constexpr (std::is_same<T, std::vector<float>>::value)
	{
		return d;//imgf.MoveData();
	}
	else
	{
		auto t = TorchUtils::make_tensor(std::move(d), //imgf.MoveData(),
			{ static_cast<int>(imgFinal.GetChannelsCount()), imgFinal.GetHeight(), imgFinal.GetWidth() });

		return t;
	}
}


template <typename T>
TENSOR_VEC_RET_VAL(T) TorchImageUtils::LoadImageAs(
	Image2d<float>& img,
	int chanCount,
	int w,
	int h)
{
	if (img.GetChannelsCount() == 0)
	{
		MY_LOG_ERROR("Failed to load image. Return zero tensor.");

		//failed to load image - return zero tensor
		if constexpr (std::is_same<T, std::vector<float>>::value)
		{
			return std::vector<float>(chanCount * h * w, 0);
		}
		else
		{
			return at::zeros({ chanCount, h, w }, at::kFloat);
		}
	}

	if (chanCount != img.GetChannelsCount())
	{
		if ((chanCount == 3) && (img.GetChannelsCount() == 1))
		{
			img = ColorSpace::ConvertGrayToRgb(img);			
		}		
		else if ((chanCount == 1) && (img.GetChannelsCount() == 3))
		{
			img = ColorSpace::ConvertRgbToGray(img);			
		}
		else
		{
			MY_LOG_ERROR("Channels count %d not supported", chanCount);
		}
	}

	auto imgFinal = ImageResize<float>::ResizeBilinear(img, ImageDimension(w, h));

	if ((imgFinal.GetWidth() != w) && (imgFinal.GetHeight() == h))
	{
		MY_LOG_ERROR("Incorrect image dimension [%d, %d]", imgFinal.GetWidth(), imgFinal.GetHeight());

		//return zero tensor
		if constexpr (std::is_same<T, std::vector<float>>::value)
		{
			return std::vector<float>(chanCount * h * w, 0);
		}
		else
		{
			return at::zeros({ chanCount, h, w }, at::kFloat);
		}
	}

	//image at input is stored as RGB/RGB/RGB
	//we need output as RRR/GGG/BBB (CHW)

	std::vector<float> d = TorchImageUtils::ImageToVector_CHW(imgFinal, {});

	if constexpr (std::is_same<T, std::vector<float>>::value)
	{
		return d; //imgFinal.MoveData();
	}
	else
	{
		auto t = TorchUtils::make_tensor(std::move(d), //imgFinal.MoveData(),
			{ static_cast<int>(imgFinal.GetChannelsCount()), imgFinal.GetHeight(), imgFinal.GetWidth() });

		return t;
	}
}


template <typename T>
static std::vector<float> TorchImageUtils::ImageToVector_CHW(
	const Image2d<T>& v,
	const MappingRange<T>& range)
{
	//image at input is stored as RGB/RGB/RGB
	//we need output as RRR/GGG/BBB (CHW)

	const auto& data = v.GetData();

	std::vector<float> d;
	d.resize(data.size());

	size_t outIndex = 0;
	float mappedVal = 0;

	for (size_t c = 0; c < v.GetChannelsCount(); c++)
	{
		for (size_t i = c; i < data.size(); i += v.GetChannelsCount())
		{
			mappedVal = Image2d<uint8_t>::MapRange(range.dataMin, range.dataMax, range.minMapTo, range.maxMapTo, data[i]);
			
			d[outIndex] = mappedVal;
			outIndex++;
		}
	}

	return d;
}

/// <summary>
/// Convert tensor to uint8_t image
/// If chanCount, w, h are -1 => dimensions taken from tensor
/// If intervalMapping is set, find min and max val in tensor
/// and use it to map [tensorMin, tensorMax] => [0, 1] => which is mapped to [0, 255]
/// If no intervalMapping is set, treat tensor as [0, 1] range (clamp values outside the range)
/// and map [0, 1] => [0, 255]
/// </summary>
/// <param name="t"></param>
/// <param name="chanCount"></param>
/// <param name="w"></param>
/// <param name="h"></param>
/// <param name="intervalMapping"></param>
/// <returns></returns>
Image2d<uint8_t> TorchImageUtils::TensorToImage(at::Tensor t,
	int chanCount,
	int w,
	int h,
	bool intervalMapping)
{
			
	if (chanCount == -1)
	{
		chanCount = t.dim();
	}

	if (w == -1)
	{
		if (t.dim() == 3)
		{
			w = t.size(1);
		}
		else 
		{
			w = t.size(0);
		}
	}

	if (h == -1)
	{
		if (t.dim() == 3)
		{
			h = t.size(2);
		}
		else
		{
			h = t.size(1);
		}
	}

	t = t.cpu().contiguous();

	if (t.dtype() != torch::kFloat32)
	{
		t = t.to(torch::kFloat32);
	}

	const float* rawData = t.const_data_ptr<float>();

	std::vector<float> flat(rawData, rawData + t.numel());

	// interval mapping / clamping
	if (intervalMapping)
	{
		float minVal = std::numeric_limits<float>::infinity();
		float maxVal = -std::numeric_limits<float>::infinity();
		for (float v : flat)
		{
			if (std::isnan(v))
			{
				continue;
			}
			if (v < minVal) minVal = v;
			if (v > maxVal) maxVal = v;
		}

		if (minVal == std::numeric_limits<float>::infinity())
		{
			// all NaNs? set to zero
			std::fill(flat.begin(), flat.end(), 0.0f);
		}
		else
		{
			float diff = maxVal - minVal;
			if (diff == 0.0f)
			{
				// if both are inside [0,1] keep values, otherwise map to zero-based
				if ((minVal < 0.0f) || (maxVal > 1.0f))
				{
					for (auto& v : flat)
					{
						v = v - minVal;
					}
					// now all zero
				}
				else
				{
					// values already in [0,1] -> keep
				}
			}
			else if ((minVal != 0.0f) && (diff != 1.0f))
			{
				for (auto& v : flat)
				{
					v = (v - minVal) / diff;
				}
			}
		}
	}
	else
	{
		// clamp to [0,1]
		for (auto& v : flat)
		{
			if (std::isnan(v)) v = 0.0f;
			else if (v < 0.0f) v = 0.0f;
			else if (v > 1.0f) v = 1.0f;
		}
	}
	
	// scale to 0..255 and convert to uint8_t
	// We'll prepare per-channel single-channel Image2d<uint8_t> and then combine with CreateFromChannels
	std::vector<Image2d<uint8_t>> channels;
	channels.reserve(chanCount);

	for (int c = 0; c < chanCount; ++c)
	{
		std::vector<uint8_t> chData;
		chData.resize(w * h);

		float* channelData = &flat[static_cast<size_t>(c)];

		for (size_t i = 0, j = 0; i < chData.size(); i++, j += chanCount)
		{
			float fv = channelData[j];
			
			// guard NaNs (converted earlier) and clamp
			if (std::isnan(fv)) fv = 0.0f;
			if (fv < 0.0f) fv = 0.0f;
			if (fv > 1.0f) fv = 1.0f;

			float scaled = 255.0f * fv + 0.5f;
			if (scaled < 0.0f) scaled = 0.0f;
			if (scaled > 255.0f) scaled = 255.0f;

			chData[i] = static_cast<uint8_t>(scaled);
		}

		// create single-channel Image2d<uint8_t> for this channel				
		channels.emplace_back(w, h, std::move(chData), ColorSpace::PixelFormat::GRAY);
	}

	// If single channel return that channel image directly
	if (chanCount == 1)
	{
		return channels[0];
	}
	else
	{
		// Combine channels into final image
		Image2d<uint8_t> out = Image2d<uint8_t>::CreateFromChannels(channels);

		return out;
	}
}




Image2d<uint8_t> TorchImageUtils::TensorsToImage(at::Tensor t,
	const TensorsToImageSettings& sets)
{	
	if (t.dim() == 3)
	{
		if (sets.colorMappingFileName.has_value())
		{
			MY_LOG_ERROR("Color pallete mapping can be used only for single channel images");
		}

		auto img = TorchImageUtils::TensorToImage(t, sets.chanCount, sets.w, sets.h, sets.intervalMapping);

		return img;
	}
	
	std::vector<std::vector<torch::Tensor>> tmpBatch;

	if (t.dim() == 4)
	{
		//input is tensor (batch, c, h, w)

		tmpBatch.push_back({});
		for (int i = 0; i < t.size(0); i++) 
		{
			tmpBatch[0].push_back(t[i]);
		}		
	}

	else if (t.dim() == 5)
	{		
		//input is tensor (batch, seq, c, h, w)

		if (sets.seqFormat == SequenceFormat::S_B)
		{
			t = t.permute({ 1, 0, 2, 3, 4 });  // (B, S, C, H, W)
		}
					
		for (int i = 0; i < t.size(0); i++) 
		{
			torch::Tensor batch = t[i];
			std::vector<torch::Tensor> tmpSeq;

			for (int j = 0; j < batch.size(0); ++j) 
			{
				tmpSeq.push_back(batch[j]);  // (C, H, W)				
			}

			tmpBatch.push_back(tmpSeq);
		}		
	}

	return TorchImageUtils::TensorsToImage(tmpBatch, sets);		
}

Image2d<uint8_t> TorchImageUtils::TensorsToImage(const std::vector<std::vector<torch::Tensor>>& t,
	const TensorsToImageSettings& sets)
{		
	const int chanCount = std::max<int>(t[0][0].size(0), sets.chanCount);
	const int w = std::max<int>(t[0][0].size(1), sets.w);
	const int h = std::max<int>(t[0][0].size(2), sets.h);

	
	int maxW = 0;
	for (int s = 0; s < t[0].size(); s++)
	{
		int seqImgW = (w + sets.borderSize) * t[0].size() + sets.borderSize;
		if (seqImgW > maxW)
		{
			maxW = seqImgW;
		}
	}

	int outputChanCount = chanCount;
	const int outputW = maxW;
	const int outputH = (h + sets.borderSize) * t.size() + sets.borderSize;


	Image2d<uint8_t> pallete;
	if (sets.colorMappingFileName.has_value())
	{
		if (chanCount > 1)
		{
			MY_LOG_ERROR("Color pallete mapping can be used only for single channel images");
		}
		else
		{
			pallete = Image2d<uint8_t>(sets.colorMappingFileName->c_str());
			outputChanCount = pallete.GetChannelsCount();
		}
	}

	const std::vector<uint8_t> defValues = std::vector<uint8_t>(outputChanCount, sets.backgroundValue);

	const auto colSpace = (outputChanCount == 1) ? ColorSpace::PixelFormat::GRAY :
		((outputChanCount == 3) ? ColorSpace::PixelFormat::RGB : ColorSpace::PixelFormat::RGBA);

	Image2d<uint8_t> newImage = Image2d<uint8_t>::CreateWithSingleValue(outputW, outputH, defValues.data(), colSpace);

	int offsetY = sets.borderSize;
	int offsetX = sets.borderSize;
	for (size_t b = 0; b < t.size(); b++)
	{
		offsetX = sets.borderSize;

		for (size_t s = 0; s < t[b].size(); s++)
		{
			auto seqImg = TorchImageUtils::TensorToImage(t[b][s], chanCount, w, h, sets.intervalMapping);

			if (sets.colorMappingFileName.has_value())
			{
				seqImg = ImageDrawing::ColorMapping(seqImg, pallete);
			}

			newImage.SetSubImage(offsetX, offsetY, seqImg);
			offsetX += w + sets.borderSize;
		}

		offsetY += h + sets.borderSize;
	}

	return newImage;
}

/// <summary>
/// takes list of torch.Tensor of size (b, seq, X)
/// and append i - th batch from each list item to rows(using maxRowsCount)
/// example return for 2 tensors:
/// [
/// 	tensors[0][batch = 0][seq = 0],
/// 	tensors[1][batch = 0][seq = 0],
/// 	tensors[0][batch = 0][seq = 1],
/// 	tensors[1][batch = 0][seq = 1],
/// 	tensors[0][batch = 0][seq = 2],
/// 	tensors[1][batch = 0][seq = 2],
/// 	....
/// 	tensors[0][batch = 1][seq = 0],
/// 	tensors[1][batch = 1][seq = 0],
/// 	tensors[0][batch = 1][seq = 1],
/// 	tensors[1][batch = 1][seq = 1],
/// 	...
/// ]
/// </summary>
/// <param name="tensors"></param>
/// <param name="maxRowsCount"></param>
/// <returns></returns>
std::vector<std::vector<torch::Tensor>> TorchImageUtils::MergeTensorsToRows(
	const std::vector<torch::Tensor>& tensors,
	int maxRowsCount)
{
	// assume all tensors have the same shape (b, seq, X)
	int seqLen = tensors[0].size(1);
	maxRowsCount = std::min<int>(tensors[0].size(0), maxRowsCount);

	std::vector<std::vector<torch::Tensor>> rows;

	for (int b = 0; b < maxRowsCount; b++) 
	{
		for (const auto& t : tensors) 
		{
			std::vector<torch::Tensor> row(seqLen);
			for (int i = 0; i < seqLen; i++) 
			{
				// take single frame [b][i] -> shape (X)
				row[i] = t[b][i];
			}
			rows.push_back(row);
		}
	}

	return rows;
}

//============================================================================================
//============================================================================================
//============================================================================================

template std::vector<float> TorchImageUtils::LoadImageAs<std::vector<float>>(
	const std::string& imgPath,
	int chanCount,
	int w,
	int h);

template torch::Tensor TorchImageUtils::LoadImageAs<torch::Tensor>(
	const std::string& imgPath,
	int chanCount,
	int w,
	int h);


template std::vector<float> TorchImageUtils::LoadImageAs<std::vector<float>>(
	Image2d<uint8_t>& img,
	int chanCount,
	int w,
	int h,
	const MappingRange<uint8_t>& range);

template torch::Tensor TorchImageUtils::LoadImageAs<torch::Tensor>(
	Image2d<uint8_t>& img,
	int chanCount,
	int w,
	int h,
	const MappingRange<uint8_t>& range);

template std::vector<float> TorchImageUtils::LoadImageAs<std::vector<float>>(
	Image2d<float>& img,
	int chanCount,
	int w,
	int h);

template torch::Tensor TorchImageUtils::LoadImageAs<torch::Tensor>(
	Image2d<float>& img,
	int chanCount,
	int w,
	int h);

template std::vector<float> TorchImageUtils::ImageToVector_CHW(
	const Image2d<uint8_t>& v,
	const MappingRange<uint8_t>& range);

template std::vector<float> TorchImageUtils::ImageToVector_CHW(
	const Image2d<float>& v,
	const MappingRange<float>& range);



