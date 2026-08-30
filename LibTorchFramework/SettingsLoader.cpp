#include "./SettingsLoader.h"

#include <Utils/CmdParser.h>
#include <Utils/Logger.h>

#include <stdexcept>

ModelSettings SettingsLoader::Load(CmdParser& cmd, const char* jsonSwitch)
{
    JsonCmdDefaults json(cmd);

    if (!json.LoadDefaultsFromJson(jsonSwitch))
    {
        throw std::runtime_error("Unable to load JSON configuration");
    }

    if (!cmd.Parse())
    {
        throw std::runtime_error("Unable to parse command line arguments");
    }

    return SettingsLoader::Load(json);
}

ModelSettings SettingsLoader::LoadFromFile(CmdParser& cmd, const char* filePath)
{
    JsonCmdDefaults json(cmd);

    if (!json.LoadDefaultsFromJsonFile(filePath))
    {
        return SettingsLoader::Load(cmd, "config");
    }

    if (!cmd.Parse())
    {
        throw std::runtime_error("Unable to parse command line arguments");
    }

    return SettingsLoader::Load(json);
}

ModelSettings SettingsLoader::Load(JsonCmdDefaults& json)
{
    ModelSettings settings;

    settings.modelId = json.GetValue<std::string>("model_id");
    settings.device = json.GetValue<std::string>("device", "gpu");

    LoadTraining(json, settings.training);
    LoadDataset(json, settings.dataset);
    LoadSnapshot(json, settings.snapshot);
    LoadDashboard(json, settings.dashboard);

    return settings;
}

void SettingsLoader::LoadTraining(const JsonCmdDefaults& json, TrainingSettings& settings)
{
    settings.epochCount = json.GetValue<int>("training.epoch_count", settings.epochCount);
    settings.numWorkers = json.GetValue<int>("training.num_workers", settings.numWorkers);
    settings.autocast = json.GetValue<bool>("training.autocast", settings.autocast);
    settings.batchSize = json.GetValue<int>("training.batch_size", settings.batchSize);
}

void SettingsLoader::LoadDataset(const JsonCmdDefaults& json, DatasetSettings& settings)
{
    settings.path = json.GetValue<std::string>("dataset.path", settings.path);

    int seed = json.GetValue<int>("dataset.seed", -1);
    settings.seed = seed >= 0 ? std::optional<int>(seed) : std::nullopt;

    int subsetSize = json.GetValue<int>("dataset.subset_size", -1);
    settings.subsetSize = subsetSize >= 0 ? std::optional<size_t>(subsetSize) : std::nullopt;

    settings.channelsCount = json.GetValue<int>("dataset.channels_count", settings.channelsCount);
    settings.width = json.GetValue<int>("dataset.width", settings.width);
    settings.height = json.GetValue<int>("dataset.height", settings.height);
    settings.prevCount = json.GetValue<int>("dataset.prev_count", settings.prevCount);
    settings.futureCount = json.GetValue<int>("dataset.future_count", settings.futureCount);
}

void SettingsLoader::LoadSnapshot(const JsonCmdDefaults& json, SnapshotSettings& settings)
{
    settings.weights = json.GetValue<std::string>("snapshot.weights", settings.weights);
    settings.path = json.GetValue<std::string>("snapshot.path", settings.path);
    settings.enableSave = json.GetValue<bool>("snapshot.enable_save", settings.enableSave);
    settings.enableLoad = json.GetValue<bool>("snapshot.enable_load", settings.enableLoad);
}

void SettingsLoader::LoadDashboard(const JsonCmdDefaults& json, DashboardSettings& settings)
{
    settings.url = json.GetValue<std::string>("dashboard.url", settings.url);
    settings.token = json.GetValue<std::string>("dashboard.token", settings.token);
}