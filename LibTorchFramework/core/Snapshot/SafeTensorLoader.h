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
};

class SafeTensorLoader
{
public:
    SafeTensorLoader() = default;
    virtual ~SafeTensorLoader() = default;

    TensorMap LoadSafetensorsSharded(const std::filesystem::path& modelDir);

    TensorMap LoadSafetensorsSharded(const std::filesystem::path& modelDir,
        std::function<std::string(const std::string&)> remap);

    
    LoadStateDictReport FillModelStateDict(
        AbstractModel& model,
        const TensorMap& mappedStateDict,
        bool strict = false);


protected:
    void MergeTensorMap(TensorMap& out, const TensorMap& add) const;
    
    TensorMap LoadFromFile(const std::filesystem::path& fileName,
        std::function<std::string(const std::string&)> remap);
};

#endif
