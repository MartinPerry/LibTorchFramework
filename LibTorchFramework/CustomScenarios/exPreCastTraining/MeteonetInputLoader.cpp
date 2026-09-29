#include "./MeteonetInputLoader.h"

#include <filesystem>

#include <Compression/3rdParty/gif_write.h>
#include <FileUtils/Reading/RawFileReader.h>
#include <Utils/Strings/StringUtils.h>

#include "../../Utils/TorchImageUtils.h"
#include "../../SettingsLoader.h"

using namespace CustomScenarios::exPreCastTraining;

MeteonetInputLoader::MeteonetInputLoader(
    RunMode type,
    std::weak_ptr<InputLoadersWrapper> parent,
    const std::string& datasetPath,
    int prevSeqLen,
    int futureSeqLen,
    const DatasetSettings& params) :
    VideoSequenceInputLoader(type, parent, datasetPath, prevSeqLen, futureSeqLen)
{   

    yearFrom = params.GetParamAs<int>("start_year", 2016);
    yearTo = params.GetParamAs<int>("end_year", 2018);    
    seqOverlap = params.GetParamAs<int>("overlap_count", sets.prevSeqLen + sets.futureSeqLen);
}


void MeteonetInputLoader::Load()
{   
    int seqLen = sets.prevSeqLen + sets.futureSeqLen;
               
    const std::vector<int> days = {
        31, 28, 31, 30, 31, 30,
        31, 31, 30, 31, 30, 31
    };

    const std::vector<int> maxMonths = {
        1, 12, 10
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
        for (int month = 1; month <= maxMonths[year - yearFrom]; ++month)
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

                    auto path = (std::filesystem::path(ymd.str()) / radarFilename);
                    
                    
                    if (std::filesystem::exists(std::filesystem::path(sets.datasetPath) / path) == false)
                    {
                        MY_LOG_ERROR("File %s not found", path.string().c_str());

                        if (allFiles.size() > 0)
                        {
                            allFiles.emplace_back(allFiles.back());
                        }
                    }
                    else
                    {
                        allFiles.emplace_back(path.string());
                    }                    
                }
            }
        }
    }

    if (allFiles.size() == 0)
    {
        MY_LOG_ERROR("No data loaded");
        return;
    }

    data.clear();
    
    for (size_t i = 0; i < allFiles.size() - seqLen; i += seqOverlap)
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

Image2d<float> MeteonetInputLoader::LoadAsImage(const std::string& p) const
{
    RawFileReader f(p.c_str());
    if (f.IsOpened() == false)
    {
        MY_LOG_ERROR("File %s not found", p.c_str());
        return Image2d<float>();
    }
    std::vector<uint8_t> buf;
    f.ReadAll(buf);
    f.Close();


    Image2d<float> img = Image2d<float>::CreateFromRawMemory(buf.data(), buf.size());

    return img;
}

std::vector<float> MeteonetInputLoader::LoadImage(const std::string& p) const
{    
    Image2d<float> img = this->LoadAsImage(p);
    if (img.GetData().size() == 0)
    {
        return std::vector<float>(sets.imgChannelsCount * sets.imgW * sets.imgH, 0.0f);
    }
    
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
    sets.intervalMapping.enabled = false;
    //sets.intervalMapping.mapRange = TorchImageUtils::MappingRange<float>();

    auto img = TorchImageUtils::TensorsToImage(seq, sets);    
    img.Save(outputName.c_str());

    auto imgs = TorchImageUtils::TensorsToImages<uint8_t>(seq, sets);

    auto w = imgs[0].GetWidth();
    auto h = imgs[0].GetHeight();

    auto gifFileName = outputName + ".gif";
    int delay = 20;
    GifWriter g = {
        .f = nullptr,
        .oldImage = nullptr,
        .firstFrame = true,
        .padding = {0}
    };

    GifBegin(&g, gifFileName.c_str(), w, h, delay);
    for (auto& gimg : imgs)
    {   
        gimg = ColorSpace::ConvertRgbToRgba(gimg, 255);

        GifWriteFrame(&g, gimg.GetData().data(), w, h, delay);        
    }
    GifEnd(&g);    

}


//=============================================================================================

#include <RasterData/OpticalFlow/Trec.h>
#include <RasterData/OpticalFlow/LucasKanade.h>
#include <RasterData/OpticalFlow/OpticalFlowBase.h>

#include <FileUtils/Writing/LZ4FileWriter.h>
#include <FileUtils/Reading/LZ4FileReader.h>


#include "../../Utils/TorchUtils.h"
#include "../../Utils/TorchImageUtils.h"

#include "../../Utils/ProgressBar.h"

std::pair<torch::Tensor, torch::Tensor> MeteonetInputLoader::LoadSequence(const SequenceInfo& si) const
{
    const size_t imgSize = sets.imgChannelsCount * sets.imgH * sets.imgW;

    std::vector<float> prev = this->CreateEmptySequence(sets.prevSeqLen);
    std::vector<float> fut = this->CreateEmptySequence(sets.futureSeqLen);

    for (int i = 0; i < static_cast<int>(si.sequenceFiles.size()); i++)
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
    
    std::filesystem::path outputDir = sets.datasetPath;
    outputDir.append("lucas_kanade");

    std::filesystem::path prevPath = sets.datasetPath;
    prevPath.append(si.sequenceFiles[sets.prevSeqLen - 2]);

    std::filesystem::path lastPath = sets.datasetPath;
    lastPath.append(si.sequenceFiles[sets.prevSeqLen - 1]);

    std::string flowFileName = prevPath.stem().string();
    flowFileName += "_";
    flowFileName += lastPath.stem().string();

    std::filesystem::path predFileName = outputDir;
    predFileName.append(flowFileName);

    std::vector<float> predData;
    Lz4FileReader lz4(predFileName.string().c_str());    
    lz4.ReadAll(predData);
    lz4.Close();

    if (predData.empty())
    {
        predData.resize(prev.size(), 0.0f);
    }

    auto tPrev = TorchUtils::make_tensor(std::move(prev),
        { sets.prevSeqLen, sets.imgChannelsCount, sets.imgH, sets.imgW });

    auto tFut = TorchUtils::make_tensor(std::move(fut),
        { sets.futureSeqLen, sets.imgChannelsCount, sets.imgH, sets.imgW });

    auto tFlow = TorchUtils::make_tensor(std::move(predData),
        { sets.futureSeqLen, sets.imgChannelsCount, sets.imgH, sets.imgW });

    auto merged = torch::cat({ tPrev, tFlow }, 0);

    return { merged, tFut };
}

void MeteonetInputLoader::PrecalcVectorField()
{
    std::filesystem::path outputDir = sets.datasetPath;
    outputDir.append("lucas_kanade");

    std::error_code ec;
    std::filesystem::create_directories(outputDir, ec);

   
    const unsigned int threadCount = std::max(1u, std::thread::hardware_concurrency());

    ProgressBar pBar(100);
    pBar.Start(data.size());

    std::atomic_size_t nextIndex = 0;
    std::mutex progressMutex;

    std::vector<std::thread> workers;
    workers.reserve(threadCount);

    for (unsigned int t = 0; t < threadCount; ++t)
    {
        workers.emplace_back([&, this]() {

                Trec::TrecSettings ts;
                ts.kernelRadius = 5;

                auto flow = std::make_shared<LucasKanade>(20);
                flow->SetWarpAlgorithm(OpticalFlowBase::WarpAlgorithm::Bicubic);

                while (true)
                {
                    const size_t index = nextIndex.fetch_add(1, std::memory_order_relaxed);
                    if (index >= data.size())
                    {
                        break;
                    }

                    const auto& d = data[index];

                    const auto& prev = d.sequenceFiles[sets.prevSeqLen - 2];
                    const auto& last = d.sequenceFiles[sets.prevSeqLen - 1];

                    std::filesystem::path prevPath = d.dirPath;
                    prevPath.append(prev);

                    std::filesystem::path lastPath = d.dirPath;
                    lastPath.append(last);

                    std::string flowFileName = prevPath.stem().string();
                    flowFileName += "_";
                    flowFileName += lastPath.stem().string();

                    std::filesystem::path predFileName = outputDir;
                    predFileName.append(flowFileName);

                    if (!std::filesystem::exists(predFileName))
                    {
                        auto tmpStart = LoadAsImage(prevPath.string());
                        auto tmpEnd = LoadAsImage(lastPath.string());

                        flow->Run(tmpEnd, tmpStart);

                        const size_t imgSize = tmpStart.GetWidth() * tmpStart.GetHeight();
                        std::vector<float> predData(sets.futureSeqLen * imgSize);

                        for (int i = 0; i < sets.futureSeqLen; ++i)
                        {
                            Image2d<float> trecRec = flow->Warp(tmpEnd, -1);

                            std::copy(
                                trecRec.GetData().begin(),
                                trecRec.GetData().end(),
                                predData.begin() + static_cast<size_t>(i) * imgSize
                            );

                            tmpEnd = std::move(trecRec);
                        }

                        Lz4FileWriter lz4(predFileName.string().c_str());
                        lz4.Write(predData);
                        lz4.Close();
                    }

                    {
                        std::lock_guard lock(progressMutex);
                        pBar.NextStep();
                    }
                }
            });
    }

    for (auto& worker : workers)
    {
        worker.join();
    }

    pBar.Finish();
}