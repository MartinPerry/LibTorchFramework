#ifndef SNAPSHOT_LOADER_H
#define SNAPSHOT_LOADER_H

class AbstractModel;
class FreezeInfo;
class PretrainedManager;

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <variant>

#include <torch/torch.h>

class SnapshotLoader 
{
public:
    explicit SnapshotLoader(const AbstractModel* model);

    bool Load(const std::variant<std::string, std::shared_ptr<PretrainedManager>>& path,
        bool forceLoad = false,
        std::shared_ptr<FreezeInfo> freezeInfo = nullptr);

private:

    bool LoadParametersFromSerialized(const std::string& path);
    bool LoadParametersFromDict(const std::string& path);
        
    void UpdateFreeze(std::shared_ptr<FreezeInfo> freezeInfo);

    const AbstractModel* model;
};

#endif