#pragma once

#include <string>
#include <optional>
#include <unordered_map>

class CmdParser;
class JsonCmdDefaults;

namespace
{
    template <typename T>
    static T GetParamAs(const std::string& key, const std::unordered_map<std::string, std::string>& params, T defaultVal = {})
    {
        auto it = params.find(key);
        if (it == params.end())
        {
            return defaultVal;
        }

        const std::string& value = it->second;

        if constexpr (std::is_same_v<T, std::string>)
        {
            return value;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return std::stod(value);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            return std::stof(value);
        }
        else if constexpr (std::is_same_v<T, int>)
        {
            return std::stoi(value);
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            if (value == "true" || value == "1")
            {
                return true;
            }

            if (value == "false" || value == "0")
            {
                return false;
            }
        }

        return defaultVal;
    }
}


struct TrainingSettings
{
    int epochCount = 100;
    int numWorkers = 4;
    bool autocast = false;
    int batchSize = 2;
    int gpuCount = 1;
};

struct DatasetSettings
{
    std::string path;
    std::optional<size_t> subsetSize;
    std::optional<int> seed = std::nullopt;
    int channelsCount = 1;
    int width = 256;
    int height = 256;
    //int prevCount = 12;
    //int futureCount = 12;

    std::unordered_map<std::string, std::string> params;

    template <typename T>
    T GetParamAs(const std::string& key, T defaultVal = {}) const
    {
        return ::GetParamAs<T>(key, this->params, defaultVal);
    }
};

struct SnapshotSettings
{
    std::string weights;
    std::string path;
    bool enableSave = true;
    bool enableLoad = false;
};

struct DashboardSettings
{
    std::string url;
    std::string token;  
    bool enableImageUpload = false;
};

struct ModelSettings
{
    std::string modelId;
    std::string device;

    TrainingSettings training;
    DatasetSettings dataset;
    SnapshotSettings snapshot;
    DashboardSettings dashboard;
};

class SettingsLoader
{
public:
    static ModelSettings Load(CmdParser& cmd, const char* jsonSwitch = "config");
    static ModelSettings LoadFromFile(CmdParser& cmd, const char* filePath);

private:

    static ModelSettings Load(JsonCmdDefaults& json);

    static void LoadTraining(const JsonCmdDefaults& json, TrainingSettings& settings);
    static void LoadDataset(const JsonCmdDefaults& json, DatasetSettings& settings);
    static void LoadSnapshot(const JsonCmdDefaults& json, SnapshotSettings& settings);
    static void LoadDashboard(const JsonCmdDefaults& json, DashboardSettings& settings);
};

