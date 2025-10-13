#ifndef SNAPSHOT_SAVER_H
#define SNAPSHOT_SAVER_H

class AbstractModel;
class FreezeInfo;
class PretrainedManager;

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <variant>

#include <torch/torch.h>

class SnapshotSaver
{
public:
    explicit SnapshotSaver(const AbstractModel* model);

    bool Save(const std::variant<std::string, std::shared_ptr<PretrainedManager>>& path);

private:

    void SaveParametersSerialized(const std::string& path);
    void SaveParametersAsDict(const std::string& path);
    
    const AbstractModel* model;
};

#endif