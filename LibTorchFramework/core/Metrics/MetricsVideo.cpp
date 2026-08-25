#include "./MetricsVideo.h"

#include <filesystem>

#include <Compression/3rdParty/gif_write.h>
#include <RasterData/Image2d.h>

#include "./PredictionEvaluators.h"

#include "../../Utils/TorchImageUtils.h"

MetricsVideo::MetricsVideo() :
    MetricsImage(MetricsType::UNKNOWN)
{
}

void MetricsVideo::Save(const std::string& filePath) const
{
    MetricsImage::Save(filePath);


    // make a local copy of images (because we will modify shapes)
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> imgList;
    imgList.reserve(images.size());
    for (auto& p : images)
    {
        imgList.emplace_back(p);
    }

    // ensure each tensor has sequence dim (unsqueeze dim=1 if needed)
    for (auto& pp : imgList)
    {
        auto& t = std::get<0>(pp);
        auto& p = std::get<1>(pp);

        if (t.dim() == 4)
        {
            // (b, c, h, w) -> add seq dim
            t = t.unsqueeze(1);
        }

        if (p.dim() == 4)
        {
            p = p.unsqueeze(1);
        }
    }


    for (size_t i = 0; i < imgList.size(); ++i)
    {
        // TorchImageUtils::MergeTensorsToRows expects vector<torch::Tensor> shaped [b, seqLen, ...]
        std::vector<torch::Tensor> toMerge = { std::get<0>(imgList[i]), std::get<1>(imgList[i]) };
        auto rows = TorchImageUtils::MergeTensorsToRows(toMerge);


        TorchImageUtils::TensorsToImageSettings sets;
        sets.borderSize = 5;
        sets.colorMappingFileName = this->colorMapping;
        sets.intervalMapping = this->intervalMapping;

        auto imgs = TorchImageUtils::TensorsToImages(rows, sets);
        std::string imgPath = this->BuildPath(filePath, static_cast<int>(i), "gif", false);

        auto w = imgs[0].GetWidth();
        auto h = imgs[0].GetHeight() * rows.size();

        std::vector<Image2d<uint8_t>> newImgs;

        int delay = 20;
        GifWriter g;
        GifBegin(&g, imgPath.c_str(), w, h, delay);
        
        size_t seqLen = rows[0].size();

        for (size_t i = 0; i < seqLen; i++)
        {
            size_t index = i;
            auto tmp = ColorSpace::ConvertRgbToRgba(imgs[i], 255);

            for (size_t r = 1; r < rows.size(); r++)
            {                                
                index += seqLen;
                tmp.AppendBottom(ColorSpace::ConvertRgbToRgba(imgs[index], 255));                
            }

            newImgs.push_back(std::move(tmp));

            GifWriteFrame(&g, newImgs.back().GetData().data(), w, h, delay);
        }
        GifEnd(&g);

    }
}

void MetricsVideo::CsiThresholds(torch::Tensor p, torch::Tensor t)
{
    //convert (B, T, C, H, W) to (B*T, C, H, W)
    //p = p.flatten(0, 1);
    //t = t.flatten(0, 1);

    //convert (B, T, C, H, W) to (B, T * C, H, W)
    p = p.reshape({ p.size(0), p.size(1) * p.size(2), p.size(3), p.size(4) });
    t = t.reshape({ t.size(0), t.size(1) * t.size(2), t.size(3), t.size(4) });

    MetricsImage::CsiThresholds(p, t);

}
