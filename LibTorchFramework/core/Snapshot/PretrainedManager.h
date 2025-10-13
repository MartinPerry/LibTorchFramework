#ifndef PRETRAINED_MANAGER_H
#define PRETRAINED_MANAGER_H

class AbstractModel;
class FreezeInfo;

#include <string>
#include <vector>
#include <optional>
#include <chrono>
#include <filesystem>
#include <memory>

class PretrainedManager 
{
public:
    explicit PretrainedManager(const std::string& directory,
        std::string snapshot = "latest",
        const std::string& prefix = "");

   
    // Config methods
    std::string GetModelDir() const;
    
    void EnableTrainingSnapshot(bool val);
    void EnableSaving(bool val);
    void EnableLoading(bool val, std::shared_ptr<FreezeInfo> freeze = nullptr);
    void EnableDeleteOldSavedFiles(bool val);
    void ClearFolder() const;
    void DeleteOldTrainingFiles(const AbstractModel* model) const;

    // File path builders
    std::string BuildFilePathForSave(const AbstractModel* model,
        const std::string& suffix,
        const std::string& ext,
        std::optional<int> epochId,
        std::optional<std::string> subDir = std::nullopt);

    std::string CreatePreTrainedWeightsPath(const AbstractModel* model);
    std::optional<std::string> GetPreTrainedWeightsPath(const AbstractModel* model) const;

    friend class SnapshotLoader;

private:
    std::vector<std::string> saveHistory;
    std::string snapshot;
    std::string prefix;
    bool trainingSnapshot = false;
    bool saveModelEnabled = true;
    bool loadModelEnabled = true;
    bool deleteOldTrainFiles = false;
    bool saveModelSummaryEnabled = true;

    std::shared_ptr<FreezeInfo> freezeInfo;
    std::filesystem::path modelsDir;

    // Helpers    
    std::string GetModelFileName(const AbstractModel* model) const;
    std::string GetTimeStampFile(const std::string& filePath, const std::tm& date) const;
    std::string AddTimeStampToFilePath(const std::string& filePath) const;
    std::string GetLatestTimeStampFile(const std::string& filePath) const;
    bool CanLoadFile(const std::string& path) const;

};

#endif