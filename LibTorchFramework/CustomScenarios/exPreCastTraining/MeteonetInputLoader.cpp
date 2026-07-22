#include "./MeteonetInputLoader.h"

#include <filesystem>

#include <FileUtils/Reading/RawFileReader.h>
#include <RasterData/Image2d.h>

#include "../../Utils/TorchImageUtils.h"

using namespace CustomScenarios::exPreCastTraining;

MeteonetInputLoader::MeteonetInputLoader(
    RunMode type,
    std::weak_ptr<InputLoadersWrapper> parent,
    const std::string& datasetPath,
    int prevSeqLen,
    int futureSeqLen) :
    VideoSequenceInputLoader(type, parent, datasetPath, prevSeqLen, futureSeqLen)
{
}


void MeteonetInputLoader::Load()
{    
    int yearFrom = 2016;
    int yearTo = 2016;
       
    const std::vector<int> days = {
        31, 28, 31, 30, 31, 30,
        31, 31, 30, 31, 30, 31
    };

    std::vector<std::string> times;
    times.reserve(24 * 12);

    for (int hour = 0; hour < 24; ++hour)
    {
        for (int minute = 0; minute < 60; minute += 5)
        {
            std::ostringstream ss;
            ss << std::setw(2) << std::setfill('0') << hour
                << std::setw(2) << std::setfill('0') << minute;
            times.push_back(ss.str());
        }
    }
       
    std::vector<std::string> allFiles;

    for (int year = yearFrom; year <= yearTo; ++year)
    {
        for (int month = 1; month <= 12; ++month)
        {
            for (int day = 1; day <= days[month - 1]; ++day)
            {
                int currentDay = day;
                
                if ((year % 4 == 0) && (month == 2))
                {
                    currentDay = day + 1;
                }

                for (const auto& time : times)
                {
                    std::ostringstream ymd;
                    ymd << std::setw(4) << std::setfill('0') << year << "/"
                        << std::setw(2) << std::setfill('0') << month << "/"
                        << std::setw(2) << std::setfill('0') << currentDay;

                    std::ostringstream ymdhm;
                    ymdhm << std::setw(4) << std::setfill('0') << year
                        << std::setw(2) << std::setfill('0') << month
                        << std::setw(2) << std::setfill('0') << currentDay
                        << time;

                    std::string radarFilename = ymdhm.str() + ".tiff";

                    allFiles.emplace_back((std::filesystem::path(ymd.str()) / radarFilename).string());
                }
            }
        }
    }

    data.clear();

    int seqLen = sets.prevSeqLen + sets.futureSeqLen;
    for (size_t i = 0; i < allFiles.size() - seqLen; i += seqLen)
    {
        auto& d = data.emplace_back(sets.datasetPath);

        for (size_t j = i; j < i + seqLen; j++)
        {           
            d.sequenceFiles.emplace_back(allFiles[j]);
        }
    }

    data = this->BuildSplits(data);

    MY_LOG_INFO("Loaded %d, dataset size: %d", static_cast<int>(this->type), this->data.size());

}

void MeteonetInputLoader::LoadSequenceFiles()
{    
}

std::vector<float> MeteonetInputLoader::LoadImage(const std::string& p) const
{
    RawFileReader f(p.c_str());
    std::vector<uint8_t> buf;
    f.ReadAll(buf);

    Image2d<uint8_t> img(256, 256, std::move(buf), ColorSpace::PixelFormat::GRAY);


    auto v = TorchImageUtils::LoadImageAs<std::vector<float>>(img,
        sets.imgChannelsCount, sets.imgW, sets.imgH);

    return v;
}

void MeteonetInputLoader::SaveSequence(size_t index, const std::string& outputName,
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