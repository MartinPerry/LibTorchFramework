#include "./SegmentationInputLoader.h"

#include <filesystem>

#include <RasterData/Image2d.h>
#include <RasterData/ImageResize.h>

#include <Utils/Logger.h>

#include "../InputLoadersWrapper.h"

#include "../../Utils/TorchUtils.h"

SegmentationInputLoader::SegmentationInputLoader(    
    RunMode type,
    std::weak_ptr<InputLoadersWrapper> parent,
    const std::string& datasetPath) :
    InputLoader(type, parent)    
{
    auto ptr = parent.lock();

    sets.datasetPath = datasetPath;
    sets.imgChannelsCount = ptr->GetShape()[0];
    sets.imgW = ptr->GetShape()[1];
    sets.imgH = ptr->GetShape()[2];
}

size_t SegmentationInputLoader::GetSize() const
{
    return this->data.size();
}

bool SegmentationInputLoader::IsNumeric(const std::string& s) const
{
    return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

void SegmentationInputLoader::Load()
{
    if (this->data.size() != 0)
    {
        //already loaded
        return;
    }

    std::vector<FileInfo> tmp;

    for (auto& dirEntry : std::filesystem::recursive_directory_iterator(sets.datasetPath))
    {
        if (dirEntry.is_directory() == false)
        {
            continue;
        }

        std::string dirId = dirEntry.path().filename().string();
        
        if (this->IsNumeric(dirId) == false)
        {
            continue;
        }

        // Iterate over files in the directory
        for (auto& file : std::filesystem::directory_iterator(dirEntry.path())) 
        {
            if (file.is_regular_file() == false)
            {
                continue;
            }

            tmp.emplace_back(file.path().string(), dirId);
        }
    }
   
    data = this->BuildSplits(tmp);

    MY_LOG_INFO("Loaded %d, dataset size: %d", static_cast<int>(this->type), this->data.size());
}

torch::Tensor SegmentationInputLoader::LoadImageAsTensor(const std::string& p, int reqChannelsCount) const
{
    Image2d<uint8_t> img = Image2d<uint8_t>(p.c_str());
    if (img.GetChannelsCount() == 0)
    {
        MY_LOG_ERROR("Failed to load image %s. Return zero tensor.", p.c_str());

        //failed to load image - return zero tensor
        return at::zeros({ reqChannelsCount, sets.imgH, sets.imgW }, at::kFloat);
    }    

    if (reqChannelsCount != img.GetChannelsCount())
    {
        if (reqChannelsCount == 3)
        {
            auto tmp = ColorSpace::ConvertToRGB(img);
            if (tmp.has_value() == false)
            {
                MY_LOG_ERROR("Failed to convert image %s. Return zero tensor.", p.c_str());

                //failed to convert image - return zero tensor
                return at::zeros({ reqChannelsCount, sets.imgH, sets.imgW }, at::kFloat);
            }
            img = *tmp;
        }
        else if (reqChannelsCount == 1)
        {
            auto tmp = ColorSpace::ConvertToGray(img);
            if (tmp.has_value() == false)
            {
                MY_LOG_ERROR("Failed to convert image %s. Return zero tensor.", p.c_str());

                //failed to convert image - return zero tensor
                return at::zeros({ reqChannelsCount, sets.imgH, sets.imgW }, at::kFloat);
            }
            img = *tmp;
        }
        else 
        {
            MY_LOG_ERROR("Channels count %d not supported", reqChannelsCount);
        }
    }

    img = ImageResize<uint8_t>::ResizeBilinear(img, ImageDimension(sets.imgW, sets.imgH));
    
    if ((img.GetWidth() != sets.imgW) && (img.GetHeight() == sets.imgH))
    {
        MY_LOG_ERROR("Incorrect image dimension [%d, %d] for %s", img.GetWidth(), img.GetHeight(), p.c_str());

        //return zero tensor
        return at::zeros({ reqChannelsCount, sets.imgH, sets.imgW }, at::kFloat);
    }

    auto imgf = img.CreateAsMapped<float>(0, 255, 0.0f, 1.0f);

    auto t = TorchUtils::make_tensor(imgf.MoveData(), 
        {static_cast<int>(img.GetChannelsCount()), img.GetHeight(), img.GetWidth()});

    return t;
}

void SegmentationInputLoader::FillData(size_t index, DataLoaderData& ld)
{    
    const auto& fi = this->data[index];
    auto maskName = fi.GetMaskFileName(sets.datasetPath);

    auto mask = this->LoadImageAsTensor(maskName, sets.maskChannelsCount);
    //create binary mask:
    //mask[mask > 0.5] = 1.
    //mask[mask <= 0.5] = 0.
    mask = torch::where(mask > 0.5, torch::ones_like(mask), torch::zeros_like(mask));

    ld.input = this->LoadImageAsTensor(fi.fn, sets.imgChannelsCount);
    ld.target = mask;    
}
