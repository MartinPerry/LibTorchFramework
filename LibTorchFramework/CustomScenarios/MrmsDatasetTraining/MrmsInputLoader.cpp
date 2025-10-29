#include "./MrmsInputLoader.h"

#include <filesystem>

#include <FileUtils/Reading/LZ4FileReader.h>
#include <RasterData/Image2d.h>

#include "../../Utils/TorchImageUtils.h"

using namespace CustomScenarios::MrmsTraining;

MrmsInputLoader::MrmsInputLoader(
    RunMode type,
    std::weak_ptr<InputLoadersWrapper> parent,
    const std::string& datasetPath,
    int prevSeqLen,
    int futureSeqLen) :
    VideoSequenceInputLoader(type, parent, datasetPath, prevSeqLen, futureSeqLen)
{
}


void MrmsInputLoader::LoadSequenceFiles()
{
    for (auto& d : data)
    {
        //tile_0.lz4
        for (int i = 0; i < sets.prevSeqLen + sets.futureSeqLen; i++)
        {
            std::string n = std::format("tile_{}.lz4", i);
            d.sequenceFiles.emplace_back(std::move(n));
        }        
    }
}

std::vector<float> MrmsInputLoader::LoadImage(const std::string& p) const
{
    Lz4FileReader f(p.c_str());
    std::vector<uint8_t> buf;
    f.ReadAll(buf);

    Image2d<uint8_t> img(256, 256, std::move(buf), ColorSpace::PixelFormat::GRAY);

    auto v = TorchImageUtils::LoadImageAs<std::vector<float>>(img, sets.imgChannelsCount, sets.imgW, sets.imgH);

    return v;
}

void MrmsInputLoader::SaveSequence(size_t index, const std::string& outputName,
    std::optional<std::string> colorMappingFileName)
{
    const auto& si = this->data[index];

    auto seqParts = this->LoadSequence(si);

    auto seq = torch::cat({ seqParts.first, seqParts.second });

    TorchImageUtils::TensorsToImageSettings sets;
    sets.borderSize = 2;
    sets.colorMappingFileName = colorMappingFileName;
         
    auto img = TorchImageUtils::TensorsToImage(seq, sets);
   
    img.Save(outputName.c_str());
}