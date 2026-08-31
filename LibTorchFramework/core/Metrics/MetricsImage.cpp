#include "./MetricsImage.h"

#include <filesystem>

#include <RasterData/Image2d.h>

#include "./PredictionEvaluators.h"

#include "../../Utils/TorchImageUtils.h"

#include "./MetricsUploader.h"

MetricsImage::MetricsImage() :
    MetricsImage(MetricsType::UNKNOWN)
{
}

MetricsImage::MetricsImage(MetricsType mType) :
    MetricsDefault(),
    mType(mType),
    keepImages(2),
    colorMapping(std::nullopt),
    iPosAll(0),
    uPosAll(0),
    iInvAll(0),
    uInvAll(0),
    runningMae(0),
    runningMse(0),
    pixelsCount(0),
    threshold(0.5f)
{                                    
}

void MetricsImage::Reset()
{
    MetricsDefault::Reset();

    images.clear();

    iPosAll = 0;
    uPosAll = 0;
    iInvAll = 0;
    uInvAll = 0;

    runningMae = 0;
    runningMse = 0;

    pixelsCount = 0;
}

void MetricsImage::SetDataMapping(const TorchImageUtils::IntervalMapping& intervalMapping)
{
    this->intervalMapping = intervalMapping;
}

void MetricsImage::SetColorMappingFileName(std::optional<std::string> colorMappingFileName)
{
    this->colorMapping = colorMappingFileName;
}

void MetricsImage::SetSavedImageCount(int c)
{
    this->keepImages = c;
}

void MetricsImage::SetCsiThresholds(const std::vector<float>& csiThresholds)
{
    this->csiThresholds = csiThresholds;

    for (const float threshold : this->csiThresholds)
    {
        confMap[threshold] = { 
            std::array<float, 3>{0.0f, 0.0f, 0.0f},
            std::array<float, 3>{0.0f, 0.0f, 0.0f},
            std::array<float, 3>{0.0f, 0.0f, 0.0f}
        };            
    }
}

std::unordered_map<std::string, float> MetricsImage::GetResultExtended() const 
{
    const float SMOOTH = 1e-6;
    auto res = MetricsDefault::GetResultExtended();

    if (pixelsCount <= 0.0)
    {
        return res;
    }

    float mse = -1.0f;
    float rmse = -1.0f;
    float mae = -1.0f;
    float psnr = -1.0f;

    if (pixelsCount > 0.0) 
    {
        mse = runningMse / pixelsCount;
        rmse = std::sqrt(mse);
        mae = runningMae / pixelsCount;
        psnr = 20.0 * std::log10(1.0 / (rmse + SMOOTH));
    }
    
    res.try_emplace("mse", mse);
    res.try_emplace("rmse", rmse);
    res.try_emplace("mae", mae);
    res.try_emplace("psnr", psnr);


    bool hasJaccardData = !(iPosAll == 0.0 && uPosAll == 0.0 && iInvAll == 0.0 && uInvAll == 0.0);
    if (hasJaccardData)
    {

        float iouPos = (iPosAll + SMOOTH) / (uPosAll + SMOOTH);
        float iouInv = (iInvAll + SMOOTH) / (uInvAll + SMOOTH);
        float macroRes = (iouPos + iouInv) / 2.0;
        float iouMicro = ((iPosAll + iInvAll) + SMOOTH) / ((uPosAll + uInvAll) + SMOOTH);

        float csi = iouPos;  // CSI = positive IoU/Jaccard

        float mcr = (uPosAll - iPosAll) / pixelsCount;
        float acc = (pixelsCount - (uPosAll - iPosAll)) / pixelsCount;

        res.try_emplace("jaccard_inverted", iouInv);
        res.try_emplace("jaccard_positive", iouPos);
        res.try_emplace("jaccard_macro", macroRes);
        res.try_emplace("jaccard_micro", iouMicro);
        res.try_emplace("csi", csi);
        res.try_emplace("mcr", mcr);
        res.try_emplace("acc", acc);
    }


    float csiMeanPool1 = 0.0f;
    float csiMeanPool4 = 0.0f;
    float csiMeanPool16 = 0.0f;

    int thresholdCount = 0;


    for (const float threshold : this->csiThresholds)
    {
        auto it = confMap.find(threshold);
        if (it == confMap.end())
        {
            continue;
        }

        auto& c = it->second;

        const float csiPool1 = this->ComputeCsi({ c[0][0], c[0][1], c[0][2] });
        const float csiPool4 = this->ComputeCsi({ c[1][0], c[1][1], c[1][2] });
        const float csiPool16 = this->ComputeCsi({ c[2][0], c[2][1], c[2][2] });

        csiMeanPool1 += csiPool1;
        csiMeanPool4 += csiPool4;
        csiMeanPool16 += csiPool16;

        ++thresholdCount;

        const std::string thresholdStr = std::to_string(threshold);

        res.try_emplace("csi_" + thresholdStr + "_pool1", csiPool1);
        res.try_emplace("csi_" + thresholdStr + "_pool4", csiPool4);
        res.try_emplace("csi_" + thresholdStr + "_pool16", csiPool16);
    }

    if (thresholdCount > 0)
    {
        res.try_emplace("csi_mean_pool1", csiMeanPool1 / static_cast<float>(thresholdCount));
        res.try_emplace("csi_mean_pool4", csiMeanPool4 / static_cast<float>(thresholdCount));
        res.try_emplace("csi_mean_pool16", csiMeanPool16 / static_cast<float>(thresholdCount));
    }

    return res;
}

std::string MetricsImage::BuildPath(const std::string& path,
    int fileIndex,
    const std::string& extension,
    bool extensionSeparateDir) const
{
    // Replace ".json" with ".{extension}"
    std::string imgPath = path;
    size_t pos = imgPath.rfind(".json");
    if (pos != std::string::npos) 
    {
        imgPath.replace(pos, 5, "." + extension);
    }

    // Extract file name
    std::filesystem::path p(imgPath);
    std::string fileName = p.filename().string();

    std::filesystem::path finalDir;
    if (extensionSeparateDir) 
    {
        // Replace filename with fileIndex, then add extension subdir
        finalDir = p.parent_path() / std::to_string(fileIndex) / extension;
    }
    else 
    {
        finalDir = p.parent_path() / std::to_string(fileIndex);
    }

    // Create directories if they don't exist
    std::filesystem::create_directories(finalDir);

    // Final path = finalDir + fileName
    std::filesystem::path finalPath = finalDir / fileName;

    return finalPath.string();
}

void MetricsImage::Save(const SaveInfo& si) const
{
    MetricsDefault::Save(si);

    
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

        auto img = TorchImageUtils::TensorsToImage(rows, sets);
        std::string imgPath = this->BuildPath(si.jsonFilePath, static_cast<int>(i), "jpg", false);
        img.Save(imgPath.c_str());

        if (dashboard != nullptr)
        {
            dashboard->UploadImage(i, imgPath, si);
        }
    }
   

}

void MetricsImage::AddImages(torch::Tensor p, torch::Tensor t)
{
    if (keepImages == 0)
    {
        return;
    }

    images.emplace_back(t.cpu(), p.cpu());
    if (static_cast<int>(images.size()) > keepImages)
    {        
        images.pop_front();
    }
}

void MetricsImage::Evaluate(torch::Tensor pred, torch::Tensor target)
{
    if (this->predEval)
    {
        pred = this->predEval->Convert(pred);
    }

    this->AddImages(pred, target);

    //todo: calculate psnr, ssim etc ?

    pixelsCount += pred.numel();

    this->RunningRmseMae(pred, target);

    //will rewrite pred and target values by threshold
    this->JaccardIndexBinary(pred, target);    

    this->CsiThresholds(pred, target);
    
}

//==================================================================================
// Calculations of metrics
//==================================================================================

void MetricsImage::RunningRmseMae(torch::Tensor p, torch::Tensor t)
{
    auto error = p - t;
    auto error2 = error * error;

    runningMae += torch::abs(error).sum().item().toFloat();
    runningMse += error2.sum().item().toFloat();
}

//==================================================================================

void MetricsImage::JaccardIndexBinary(torch::Tensor p, torch::Tensor t, bool mergeBatches)
{
    //if mergeBatches is True, results are for all batches together
    //if mergeBatches is False, each batch has its own result

    // BATCH x(....)->each batch is linearized

    auto batchSize = p.size(0);

    auto pThreshold = (p > threshold).to(torch::kUInt8);
    auto tThreshold = (t > threshold).to(torch::kUInt8);

    pThreshold = pThreshold.view({ batchSize, -1 });
    tThreshold = tThreshold.view({ batchSize, -1 });

    float iPosAll_local = 0.0f;
    float uPosAll_local = 0.0f;
    float iInvAll_local = 0.0f;
    float uInvAll_local = 0.0f;
    std::tie(iPosAll_local, uPosAll_local, iInvAll_local, uInvAll_local) = CalcIntersectUnions(pThreshold, tThreshold, mergeBatches);

    iPosAll += iPosAll_local;
    uPosAll += uPosAll_local;
    iInvAll += iInvAll_local;
    uInvAll += uInvAll_local;
}

std::tuple<float, float, float, float> MetricsImage::CalcIntersectUnions(torch::Tensor p, torch::Tensor t, bool mergeBatches) const 
{
    int64_t batchSize = p.size(0);

    double iPosAll_local = 0.0;
    double uPosAll_local = 0.0;
    double iInvAll_local = 0.0;
    double uInvAll_local = 0.0;

    if (mergeBatches) 
    {
        for (int64_t b = 0; b < batchSize; ++b) 
        {
            auto p_b = p[b].unsqueeze(0);
            auto t_b = t[b].unsqueeze(0);
            auto iPos_uPos = Iou(p_b, t_b);
            auto iInv_uInv = IouInverse(p_b, t_b);

            iPosAll_local += iPos_uPos.first.sum().item().toFloat();
            uPosAll_local += iPos_uPos.second.sum().item().toFloat();
            iInvAll_local += iInv_uInv.first.sum().item().toFloat();
            uInvAll_local += iInv_uInv.second.sum().item().toFloat();
        }
    }
    else 
    {
        auto iPos_uPos = Iou(p, t);
        auto iInv_uInv = IouInverse(p, t);
        iPosAll_local = iPos_uPos.first.sum().item().toFloat();
        uPosAll_local = iPos_uPos.second.sum().item().toFloat();
        iInvAll_local = iInv_uInv.first.sum().item().toFloat();
        uInvAll_local = iInv_uInv.second.sum().item().toFloat();
    }

    return std::make_tuple(iPosAll_local, uPosAll_local, iInvAll_local, uInvAll_local);
}

std::pair<torch::Tensor, torch::Tensor> MetricsImage::IouInverse(const torch::Tensor& p, const torch::Tensor& t) const 
{
    auto predInv = 1 - p;
    auto targetInv = 1 - t;
    return Iou(predInv, targetInv);
}

std::pair<torch::Tensor, torch::Tensor> MetricsImage::Iou(const torch::Tensor& p, const torch::Tensor& t) const 
{
    // pred and target are expected to be byte/uint8 or boolean tensors
    auto intersection = (p & t).to(torch::kFloat32);
    auto uni = (p | t).to(torch::kFloat32);

    // sum along dimension 1 (each row)
    auto intersection_sum = intersection.sum(1);
    auto union_sum = uni.sum(1);

    return { intersection_sum, union_sum };
}

//==================================================================================

void MetricsImage::CsiThresholds(torch::Tensor p, torch::Tensor t)
{
    for (const float threshold : this->csiThresholds)
    {
        auto& c = confMap[threshold];

        auto t0 = ComputeHitsMissesFas(p, t, threshold);
        auto t1 = ComputePooledConfusion(p, t, threshold, 4, PoolType::MAX);
        auto t2 = ComputePooledConfusion(p, t, threshold, 16, PoolType::MAX);

        c[0][0] += std::get<0>(t0);
        c[0][1] += std::get<1>(t0);
        c[0][2] += std::get<2>(t0);

        c[1][0] += std::get<0>(t1);
        c[1][1] += std::get<1>(t1);
        c[1][2] += std::get<2>(t1);

        c[2][0] += std::get<0>(t2);
        c[2][1] += std::get<1>(t2);
        c[2][2] += std::get<2>(t2);
    }
}

std::tuple<float, float, float> MetricsImage::ComputeHitsMissesFas(const torch::Tensor& p, const torch::Tensor& t, double threshold) const
{
    torch::Tensor gtsBin = (t >= threshold).to(torch::kFloat32);
    torch::Tensor predsBin = (p >= threshold).to(torch::kFloat32);

    torch::Tensor hits = torch::sum(gtsBin * predsBin);
    torch::Tensor misses = torch::sum(gtsBin * (1.0 - predsBin));
    torch::Tensor fas = torch::sum((1.0 - gtsBin) * predsBin);

    return { hits.item<float>(), misses.item<float>(), fas.item<float>() };
}

std::tuple<float, float, float> MetricsImage::ComputePooledConfusion(const torch::Tensor& p, const torch::Tensor& t,
    double threshold, int64_t poolSize, PoolType mode) const
{   
    // Python:
    // stride = math.ceil(pool_size / 4)
    //
    // Integer ceiling division:
    // ceil(poolSize / 4) = (poolSize + 3) / 4
    const int64_t stride = (poolSize + 3) / 4;

    torch::Tensor pooledGts;
    torch::Tensor pooledPreds;

    if (mode == PoolType::MAX)
    {
        pooledGts = torch::max_pool2d(
            t,
            { poolSize, poolSize },
            { stride, stride });

        pooledPreds = torch::max_pool2d(
            p,
            { poolSize, poolSize },
            { stride, stride });
    }
    else
    {
        pooledGts = torch::avg_pool2d(
            t,
            { poolSize, poolSize },
            { stride, stride });

        pooledPreds = torch::avg_pool2d(
            p,
            { poolSize, poolSize },
            { stride, stride });
    }

    return this->ComputeHitsMissesFas(
        pooledGts,
        pooledPreds,
        threshold);
}

float MetricsImage::ComputeCsi(std::tuple<float, float, float> hitMissFas) const
{
    auto [hits, misses, false_alarms] = hitMissFas;

    auto csi = hits / (hits + misses + false_alarms + 1e-6);
    return csi;
}