#include "./SegmentationInputLoader.h"

#include <filesystem>

#include <Utils/Logger.h>

#include "../InputLoadersWrapper.h"

#include "../../Utils/TorchUtils.h"
#include "../../Utils/TorchImageUtils.h"

//============================================
// Data in dataset dir are expected to be in format:
// -----
// [dir] image_id -> list of images
// [file] image_id.png -> mask
// ------
// image_id is id of image - in its folder, there are multiple versions of the
// same image (eg. webcam images, augmented images etc)
// image_id.png is mask for the image
//============================================

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


void SegmentationInputLoader::FillData(size_t index, DataLoaderData& ld)
{    
    const auto& fi = this->data[index];
    auto maskName = fi.GetMaskFileName(sets.datasetPath);

    auto mask = TorchImageUtils::LoadImageAs<torch::Tensor>(maskName, sets.maskChannelsCount, sets.imgW, sets.imgH);    
    //create binary mask:
    //mask[mask > 0.5] = 1.
    //mask[mask <= 0.5] = 0.
    mask = torch::where(mask > 0.5, torch::ones_like(mask), torch::zeros_like(mask));

    ld.input = TorchImageUtils::LoadImageAs<torch::Tensor>(fi.fn, sets.imgChannelsCount, sets.imgW, sets.imgH);        
    ld.target = mask;     
}
