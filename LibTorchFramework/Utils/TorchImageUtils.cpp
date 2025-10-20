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

#include <RasterData/Colors/ColorSpace.h>


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
			else
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

		size_t base = static_cast<size_t>(c) * w * h;

		for (size_t i = 0; i < chData.size(); ++i)
		{
			float fv = flat[base + i];
			
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
		Image2d<uint8_t> imCh(w, h, chData, ColorSpace::PixelFormat::GRAY);
		channels.push_back(std::move(imCh));
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
	TorchImageUtils::SequenceFormat seqFormat,
	int chanCount,
	int w,
	int h,
	int borderSize,
	uint8_t backgroundValue,
	bool intervalMapping)
{	
	if (t.dim() == 3)
	{
		return TorchImageUtils::TensorToImage(t, chanCount, w, h, intervalMapping);
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

		if (seqFormat == SequenceFormat::S_B) 
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

	return TorchImageUtils::TensorsToImage(tmpBatch,
		seqFormat,
		chanCount, w, h,
		borderSize,
		backgroundValue,
		intervalMapping);
}

Image2d<uint8_t> TorchImageUtils::TensorsToImage(const std::vector<std::vector<torch::Tensor>>& t,
	TorchImageUtils::SequenceFormat seqFormat,
	int chanCount,
	int w,
	int h,
	int borderSize,
	uint8_t backgroundValue,
	bool intervalMapping)
{
	
	int totalW = w;
	int totalH = h;

	chanCount = std::max<int>(t[0][0].size(0), chanCount);
	w = std::max<int>(t[0][0].size(1), w);
	h = std::max<int>(t[0][0].size(2), h);


	int maxW = 0;
	for (int s = 0; s < t[0].size(); s++)
	{
		int seqImgW = (w + borderSize) * t[0].size() + borderSize;
		if (seqImgW > maxW)
		{
			maxW = seqImgW;
		}
	}

	totalW = maxW;
	totalH = (h + borderSize) * t.size() + borderSize;


	std::vector<uint8_t> defValues = std::vector<uint8_t>(chanCount, backgroundValue);

	auto colSpace = (chanCount == 1) ? ColorSpace::PixelFormat::GRAY :
		((chanCount == 3) ? ColorSpace::PixelFormat::RGB : ColorSpace::PixelFormat::RGBA);
	Image2d<uint8_t> newImage = Image2d<uint8_t>::CreateWithSingleValue(totalW, totalH, defValues.data(), colSpace);

	int offsetY = borderSize;
	int offsetX = borderSize;
	for (size_t b = 0; b < t.size(); b++)
	{
		offsetX = borderSize;

		for (size_t s = 0; s < t[b].size(); s++)
		{
			auto seqImg = TorchImageUtils::TensorToImage(t[b][s], chanCount, w, h, intervalMapping);
			newImage.SetSubImage(offsetX, offsetY, seqImg);
			offsetX += w + borderSize;
		}

		offsetY += h + borderSize;
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











