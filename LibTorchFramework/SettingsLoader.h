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
    std::optional<int> subsetSize;
    int channelsCount = 1;
    int width = 256;
    int height = 256;
    int prevCount = 12;
    int futureCount = 12;
};

struct SnapshotSettings
{
    std::string path;
    bool enableSave = true;
    bool enableLoad = false;
};

struct ModelSettings
{
    std::string modelId;
    TrainingSettings training;
    DatasetSettings dataset;
    SnapshotSettings snapshot;
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
};

