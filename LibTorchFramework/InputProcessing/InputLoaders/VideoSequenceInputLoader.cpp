#include "./VideoSequenceInputLoader.h"

#include <filesystem>

#include <Utils/Logger.h>

#include "../InputLoadersWrapper.h"

#include "../../Utils/TorchUtils.h"
#include "../../Utils/TorchImageUtils.h"

//============================================
// Data in dataset dir are expected to be in format:
// -----
// [dir] sequence_id -> list of images in sequence
// ------
// sequence_id is id of sequence - in its folder, there are multiple images that builds the sequence
//============================================


VideoSequenceInputLoader::VideoSequenceInputLoader(
    RunMode type,
    std::weak_ptr<InputLoadersWrapper> parent,
    const std::string& datasetPath,
    int prevSeqLen,
    int futureSeqLen) :
    InputLoader(type, parent)
{
    auto ptr = parent.lock();

    sets.datasetPath = datasetPath;
    sets.imgChannelsCount = ptr->GetShape()[0];
    sets.imgW = ptr->GetShape()[1];
    sets.imgH = ptr->GetShape()[2];
    sets.prevSeqLen = prevSeqLen;
    sets.futureSeqLen = futureSeqLen;
}

size_t VideoSequenceInputLoader::GetSize() const
{
    return this->data.size();
}


void VideoSequenceInputLoader::Load()
{
    if (this->data.size() != 0)
    {
        //already loaded
        return;
    }

    std::vector<SequenceInfo> tmp;

    for (auto& dirEntry : std::filesystem::recursive_directory_iterator(sets.datasetPath))
    {
        if (dirEntry.is_directory() == false)
        {
            continue;
        }

        std::string dirPath = dirEntry.path().string();

        tmp.emplace_back(dirPath);
    }

    data = this->BuildSplits(tmp);

    this->LoadSequenceFiles();

    MY_LOG_INFO("Loaded %d, dataset size: %d", static_cast<int>(this->type), this->data.size());
}

void VideoSequenceInputLoader::LoadSequenceFiles()
{
    for (auto& d : data)
    {
        for (auto& sequenceFile : std::filesystem::recursive_directory_iterator(d.dirPath))
        {
            d.sequenceFiles.emplace_back(sequenceFile.path().filename().string());
        }
    }
}

std::vector<float> VideoSequenceInputLoader::CreateEmptySequence(int seqLen) const
{
    return std::vector<float>(seqLen * sets.imgChannelsCount * sets.imgH * sets.imgW, 0.0f);
}

std::vector<float> VideoSequenceInputLoader::LoadImage(const std::string& p) const
{    
    return TorchImageUtils::LoadImageAs<std::vector<float>>(p, sets.imgChannelsCount, sets.imgW, sets.imgH);
}

std::pair<torch::Tensor, torch::Tensor> VideoSequenceInputLoader::LoadSequence(const SequenceInfo& si) const
{
    const size_t imgSize = sets.imgChannelsCount * sets.imgH * sets.imgW;

    std::vector<float> prev = this->CreateEmptySequence(sets.prevSeqLen);
    std::vector<float> fut = this->CreateEmptySequence(sets.futureSeqLen);

    for (int i = 0; i < si.sequenceFiles.size(); i++)
    {
        std::string imgPath = si.dirPath;
        imgPath += "/";
        imgPath += si.sequenceFiles[i];

        auto img = this->LoadImage(imgPath);

        if (i < sets.prevSeqLen)
        {          
            std::copy(img.begin(), img.end(), prev.begin() + i * imgSize);
            //prev.insert(prev.end(),
            //    std::make_move_iterator(img.begin()),
            //    std::make_move_iterator(img.end()));
        }
        else
        {
            std::copy(img.begin(), img.end(), fut.begin() + (i - sets.prevSeqLen) * imgSize);
            //fut.insert(fut.end(),
            //    std::make_move_iterator(img.begin()),
            //    std::make_move_iterator(img.end()));
        }
    }

    auto tPrev = TorchUtils::make_tensor(std::move(prev),
        { sets.prevSeqLen, sets.imgChannelsCount, sets.imgH, sets.imgW });

    auto tFut = TorchUtils::make_tensor(std::move(fut),
        { sets.futureSeqLen, sets.imgChannelsCount, sets.imgH, sets.imgW });

    return { tPrev, tFut };
}

void VideoSequenceInputLoader::FillData(size_t index, DataLoaderData& ld)
{
    const auto& si = this->data[index];

    auto seq = this->LoadSequence(si);
        
    ld.input = std::get<0>(seq);
    ld.target = std::get<1>(seq);

}
