#include "./SnapshotLoader.h"

#include <unordered_set>

#include <FileUtils/Reading/RawFileReader.h>
#include <Utils/Logger.h>

#include "../AbstractModel.h"

#include "./FreezeInfo.h"
#include "./PretrainedManager.h"

SnapshotLoader::SnapshotLoader(const AbstractModel* model)  :
    model(model) 
{
}


bool SnapshotLoader::Load(const std::variant<std::string, std::shared_ptr<PretrainedManager>>& pathVariant,
    bool forceLoad,
    std::shared_ptr<FreezeInfo> freezeInfo)
{

    std::string path;
    std::shared_ptr<FreezeInfo> fi = freezeInfo;

    if (std::holds_alternative<std::shared_ptr<PretrainedManager>>(pathVariant))
    {
        std::shared_ptr<PretrainedManager> pm = std::get<std::shared_ptr<PretrainedManager>>(pathVariant);
        path = pm->GetPreTrainedWeightsPath(model).value_or("");
        fi = pm->freezeInfo; // copy freeze info
    }
    else 
    {
        path = std::get<std::string>(pathVariant);
    }

    if (path.empty()) 
    {
        MY_LOG_INFO("No trained model will be loaded");
        return false;
    }

    MY_LOG_INFO("Loading trained model from %s", path.c_str());

    this->LoadParametersFromSerialized(path);

    //this->LoadParametersFromDict(path);
                
    this->UpdateFreeze(freezeInfo);

    return true;
}

void SnapshotLoader::LoadParametersFromSerialized(const std::string& path)
{
    torch::serialize::InputArchive archive;
    archive.load_from(path);

    // Read each parameter we have in model (safe: read only those present)
    const torch::OrderedDict<std::string, at::Tensor>& modelParams = model->named_parameters();

    for (const auto& p : modelParams)
    {
        const std::string name = p.key();
        at::Tensor tmp;
        try 
        {
            archive.read(name, tmp); // tmp will be CPU tensor
            
            // Ensure same dtype/device and copy into parameter
            if (tmp.defined()) 
            {
                // If model param is on CUDA, copy to that device
                at::Tensor target = p.value();
                tmp = tmp.to(target.device()).to(target.dtype());
                torch::NoGradGuard no_grad;
                p.value().copy_(tmp);
            }
        }
        catch (const c10::Error& e) 
        {
            // parameter not found in archive or other libtorch error
            MY_LOG_WARNING("Could not read parameter '%s' from archive: %s", name.c_str(), e.what());
        }
        catch (const std::exception& e) 
        {
            MY_LOG_ERROR("Error reading '%s': %s", name.c_str(), e.what());
        }
    }

}

/// <summary>
/// https://github.com/pytorch/pytorch/issues/36577
/// </summary>
/// <param name="ptPath"></param>
void SnapshotLoader::LoadParametersFromDict(const std::string& path)
{
    std::vector<char> f;

    RawFileReader rf(path.c_str(), "rb");
    rf.ReadAll(f);
    rf.Close();
    
    c10::Dict<c10::IValue, c10::IValue> weights = torch::pickle_load(f).toGenericDict();
    
    const torch::OrderedDict<std::string, at::Tensor>& modelParams = model->named_parameters();
    
    std::vector<std::string> paramNames;
    for (auto const& w : modelParams)
    {
        paramNames.push_back(w.key());
    }

    torch::NoGradGuard no_grad;
    for (auto const& w : weights) 
    {
        std::string name = w.key().toStringRef();
        torch::Tensor param = w.value().toTensor();

        if (std::find(paramNames.begin(), paramNames.end(), name) != paramNames.end())
        {
            modelParams.find(name)->copy_(param);
        }
        else 
        {
            MY_LOG_WARNING("[%s] does not exist among model parameters", name.c_str());
        }
    }
}


void SnapshotLoader::UpdateFreeze(std::shared_ptr<FreezeInfo> freezeInfo)
{
    if (freezeInfo == nullptr)
    {
        return;
    }

    const torch::OrderedDict<std::string, at::Tensor>& params = model->named_parameters();

    std::vector<std::string> freezedParts;
    for (auto& kv : params)
    {
        if (freezeInfo->enabled && freezeInfo->CanFreeze(kv.key()))
        {
            freezedParts.push_back(kv.key());
        }
    }

    if (freezedParts.empty())
    {
        return;
    }

    for (auto& kv : params) 
    {        
        if (std::find(freezedParts.begin(), freezedParts.end(), kv.key()) != freezedParts.end()) 
        {
            MY_LOG_INFO("Freezing: %s", kv.key().c_str());
            kv.value().set_requires_grad(false);
        }
    }
}