#pragma once

#include <string>
#include <optional>

class CmdParser;
class JsonCmdDefaults;

struct TrainingSettings
{
    int epochCount = 100;
    int numWorkers = 4;
    bool autocast = false;
    int batchSize = 2;
};

struct DatasetSettings
{
    std::string path;
    std::optional<size_t> subsetSize;
    std::optional<int> seed = std::nullopt;
    int channelsCount = 1;
    int width = 256;
    int height = 256;
    int prevCount = 12;
    int futureCount = 12;
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

