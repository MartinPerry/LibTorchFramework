#include "./PretrainedManager.h"

#include "PretrainedManager.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <algorithm>

#include <Utils/Logger.h>

#include "../AbstractModel.h"

#include "./FreezeInfo.h"

//==============================================================
//  Utility functions
//==============================================================

std::time_t to_time_t(std::filesystem::file_time_type ft)
{
    using namespace std::chrono;

    // Convert from file_time_type (implementation-defined) to system_clock
    auto sctp = time_point_cast<system_clock::duration>(
        ft - std::filesystem::file_time_type::clock::now() + system_clock::now()
    );

    return system_clock::to_time_t(sctp);
}

//==============================================================
//  PretrainedManager Implementation
//==============================================================

PretrainedManager::PretrainedManager(const std::string& directory, 
    std::string snapshot, const std::string& prefix) : 
    snapshot(snapshot), 
    prefix(prefix), 
    modelsDir(directory),
    freezeInfo(nullptr)
{
    std::filesystem::create_directories(modelsDir);
    if (snapshot != "latest")
    {
        MY_LOG_INFO("PretrainedManager - direct file path, disabling saving by default");
        this->EnableSaving(false);
    }
}


std::string PretrainedManager::GetModelDir() const 
{
    return modelsDir.string();
}

void PretrainedManager::EnableTrainingSnapshot(bool val) 
{ 
    trainingSnapshot = val; 
}

void PretrainedManager::EnableSaving(bool val) 
{ 
    saveModelEnabled = val; 
}

void PretrainedManager::EnableLoading(bool val, std::shared_ptr<FreezeInfo> freeze)
{
    loadModelEnabled = val;
    freezeInfo = freeze;
}

void PretrainedManager::EnableDeleteOldSavedFiles(bool val) 
{ 
    deleteOldTrainFiles = val; 
}

void PretrainedManager::ClearFolder()  const
{
    MY_LOG_INFO("Clearing folder");
    for (const auto& entry : std::filesystem::directory_iterator(modelsDir)) 
    {
        std::filesystem::remove_all(entry);
    }
}

void PretrainedManager::DeleteOldTrainingFiles(const AbstractModel* model) const
{
    if (!deleteOldTrainFiles)
    {
        return;
    }
    MY_LOG_INFO("Deleting old training files");

    auto lastPath = this->GetPreTrainedWeightsPath(model);
    if (lastPath.has_value() == false)
    {
        return;
    }

    std::filesystem::path lastFile(lastPath.value());
    std::string lastStem = lastFile.stem().string();

    for (const auto& pathStr : saveHistory) 
    {
        std::filesystem::path p(pathStr);
        if (p.stem().string().find(lastStem) == 0)
        {
            continue;
        }
        try
        {
            std::filesystem::remove(p);
            auto jsonPath = p;
            jsonPath.replace_extension(".json");
            std::filesystem::remove(jsonPath);
        }
        catch (...) 
        {
            continue;
        }
    }
}

std::string PretrainedManager::BuildFilePathForSave(const AbstractModel* model,
    const std::string& suffix,
    const std::string& ext,
    std::optional<int> epochId,
    std::optional<std::string> subDir) 
{
    std::filesystem::path outputDir = modelsDir;
    if (subDir.has_value()) 
    {
        outputDir /= subDir.value();
        std::filesystem::create_directories(outputDir);
    }

    std::string fileName = this->GetModelFileName(model);

    if (snapshot == "latest") 
    {
        fileName = this->AddTimeStampToFilePath(fileName);
    }
    else 
    {
        fileName = snapshot;
    }    

    if (!suffix.empty())
    {
        fileName += "_" + suffix;
    }
    if (epochId.has_value())
    {
        fileName += "_" + std::to_string(epochId.value());
    }

    std::string filePath = (outputDir / (fileName + "." + ext)).string();
    saveHistory.push_back(filePath);
    return filePath;
}

std::string PretrainedManager::CreatePreTrainedWeightsPath(const AbstractModel* model)
{
    return BuildFilePathForSave(model, "", "bin", std::nullopt);
}

std::optional<std::string> PretrainedManager::GetPreTrainedWeightsPath(const AbstractModel* model) const
{
    if (!loadModelEnabled)
    {
        return std::nullopt;
    }
    std::string wp = "";

    std::string fileName = this->GetModelFileName(model);
    std::filesystem::path finalPath = modelsDir / (fileName + ".bin");

    if (snapshot == "latest")
    {
        wp = this->GetLatestTimeStampFile(finalPath.string());
    }
    else
    {
        wp = finalPath.string();
    }

    return this->CanLoadFile(wp) ? std::make_optional(wp) : std::nullopt;
}

std::string PretrainedManager::GetModelFileName(const AbstractModel* model) const
{    
    std::string fileName = (model != nullptr) ? model->GetName() : "";
    if (!prefix.empty())
    {
        fileName = prefix + "_" + fileName;
    }
    return fileName;
}

std::string PretrainedManager::GetTimeStampFile(const std::string& filePath, const std::tm& date) const 
{
    std::ostringstream oss;
    oss << std::put_time(&date, "%Y_%m_%d_%H_%M");
    auto pos = filePath.find_last_of('.');
    if (pos == std::string::npos)
    {
        return filePath + "_" + oss.str();
    }
    return filePath.substr(0, pos) + "_" + oss.str() + filePath.substr(pos);
}

std::string PretrainedManager::AddTimeStampToFilePath(const std::string& filePath) const 
{
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&t);
    return GetTimeStampFile(filePath, tm);
}

std::string PretrainedManager::GetLatestTimeStampFile(const std::string& filePath) const 
{
    std::filesystem::path basePath(filePath);
    std::string stem = basePath.stem().string();
    std::string ext = basePath.extension().string();
    std::string patternPrefix = stem + "_";

    std::filesystem::path parent = basePath.parent_path();
    std::string bestFile = filePath;
    std::time_t bestTime = 0;

    for (const auto& entry : std::filesystem::directory_iterator(parent)) 
    {
        std::string name = entry.path().filename().string();
        if (name.find(patternPrefix) != 0)
        {
            continue;
        }

        auto t = std::filesystem::last_write_time(entry);
        auto sysTime = to_time_t(t);
        if (sysTime > bestTime) 
        {
            bestTime = sysTime;
            bestFile = entry.path().string();
        }
    }
    return bestFile;
}

bool PretrainedManager::CanLoadFile(const std::string& path) const 
{
    return std::filesystem::exists(path);
}