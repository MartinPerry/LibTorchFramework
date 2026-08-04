#ifndef SAFE_TENSOR_LOADER_H
#define SAFE_TENSOR_LOADER_H

class AbstractModel;

#include <filesystem>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

using TensorMap = std::unordered_map<std::string, torch::Tensor>;

struct LoadStateDictReport
{
    std::vector<std::string> Missing;
    std::vector<std::string> Unexpected;
    std::vector<std::string> Loaded;
};

class SafeTensorLoader
{
public:
    SafeTensorLoader() = default;
    virtual ~SafeTensorLoader() = default;
    
    TensorMap LoadSafetensors(const std::filesystem::path& dataPath);

    TensorMap LoadSafetensors(const std::filesystem::path& dataPath,
        std::function<std::string(const std::string&)> remapName);

    LoadStateDictReport LoadModel(const std::filesystem::path& dataPath,
        AbstractModel& model,
        bool strict = false,
        std::function<std::string(const std::string&)> remapName = nullptr);

    
    LoadStateDictReport FillModelStateDict(
        AbstractModel& model,
        const TensorMap& mappedStateDict,
        bool strict = false);
   

protected:
    void MergeTensorMap(TensorMap& out, const TensorMap& add) const;
    
    
    std::vector<std::filesystem::path> LoadShardsFileNames(const std::filesystem::path& modelDir);
};

#endif
