#include "./EncoderDecoderInputLoader.h"

#include <filesystem>

#include <Utils/Logger.h>

#include "../InputLoadersWrapper.h"

#include "../../Utils/TorchUtils.h"
#include "../../Utils/TorchImageUtils.h"

//============================================
// Data in dataset dir are expected to be in format:
// -----
// [dir] list of images
// ------
//============================================

EncoderDecoderInputLoader::EncoderDecoderInputLoader(
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

size_t EncoderDecoderInputLoader::GetSize() const
{
    return this->data.size();
}


void EncoderDecoderInputLoader::Load()
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

torch::Tensor EncoderDecoderInputLoader::LoadImageAsTensor(const std::string& p, int reqChannelsCount) const
{
    return TorchImageUtils::LoadImageAs<torch::Tensor>(p, reqChannelsCount, sets.imgW, sets.imgH);
}

void EncoderDecoderInputLoader::FillData(size_t index, DataLoaderData& ld)
{
    const auto& fi = this->data[index];
    auto maskName = fi.GetMaskFileName(sets.datasetPath);
    
    ld.input = this->LoadImageAsTensor(fi.fn, sets.imgChannelsCount);
    ld.target = ld.input;
}
