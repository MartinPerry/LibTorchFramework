#include "./SnapshotLoader.h"

#include <unordered_set>
#include <string>
#include <regex>

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

    MY_LOG_INFO("Loading serialized trained model from %s", path.c_str());

    if (this->LoadParametersFromSerialized(path) == false)
    {
        MY_LOG_INFO("Loading pickled trained model from %s", path.c_str());

        //try to load from pickled pytorch
        this->LoadParametersFromDict(path);        
    }
                
    this->UpdateFreeze(freezeInfo);

    return true;
}

bool SnapshotLoader::LoadParametersFromSerialized(const std::string& path)
{
    
    torch::serialize::InputArchive archive;
    try
    {
        archive.load_from(path);
    }
    catch (const c10::Error& e)
    {
        // parameter not found in archive or other libtorch error
        MY_LOG_ERROR("Failed to load InputArchive: %s", path.c_str(), e.what());
        return false;
    }
    catch (const std::exception& e)
    {
        MY_LOG_ERROR("Failed to load InputArchive: %s", path.c_str(), e.what());
        return false;
    }
    
    // Read each parameter we have in model (safe: read only those present)
    torch::OrderedDict<std::string, at::Tensor> modelParams = model->named_parameters();

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

    return true;
}

/// <summary>
/// To use this, pytorch muset save with:
/// 
/// torch.save(checkpoint['state_dict'], "weights.pth")
/// or
/// torch.save(model.state_dict(), "weights.pth")
/// 
/// https://github.com/pytorch/pytorch/issues/36577
/// </summary>
/// <param name="ptPath"></param>
bool SnapshotLoader::LoadParametersFromDict(const std::string& path)
{
    std::vector<char> f;

    RawFileReader rf(path.c_str(), "rb");
    rf.ReadAll(f);
    rf.Close();
        
    torch::IValue pickle;
        
    try
    {
        pickle = torch::pickle_load(f);
    }
    catch (const c10::Error& e)
    {
        // parameter not found in archive or other libtorch error
        MY_LOG_ERROR("Failed to pickle_load: %s", path.c_str(), e.what());
        return false;
    }
    catch (const std::exception& e)
    {
        MY_LOG_ERROR("Failed to pickle_load: %s", path.c_str(), e.what());
        return false;
    }

    if (pickle.isGenericDict() == false)
    {
        return false;
    }

    c10::Dict<c10::IValue, c10::IValue> weights = pickle.toGenericDict();
    
    torch::OrderedDict<std::string, at::Tensor> modelParams = model->named_parameters();
    
    std::unordered_set<std::string> paramNames;
    for (auto const& w : modelParams)
    {
        paramNames.emplace(w.key());
    }

    torch::NoGradGuard no_grad;
    for (auto const& w : weights) 
    {
        std::string name = w.key().toStringRef();
        torch::Tensor param = w.value().toTensor();
        
        if (paramNames.find(name) != paramNames.end())
        {            
            modelParams.find(name)->copy_(param);
        }
        else 
        {
            //try to replace ".[number]. with .seq.[number].
            //since in pyhton some modules can inherit from sequential module
            //in our code, we rewrite it with 
            //torch::nn::Sequential seq;

            std::regex pattern(R"(\.(\d+)\.)");
            std::string replacement = ".seq.$1.";
            auto updatedName = std::regex_replace(name, pattern, replacement);

            if (paramNames.find(updatedName) != paramNames.end())
            {
                modelParams.find(updatedName)->copy_(param);
            }
            else
            {
                MY_LOG_WARNING("[%s] does not exist among model parameters", name.c_str());
            }
        }
    }

    return true;
}

void SnapshotLoader::UpdateFreeze(std::shared_ptr<FreezeInfo> freezeInfo)
{
    if (freezeInfo == nullptr)
    {
        return;
    }

    torch::OrderedDict<std::string, at::Tensor> params = model->named_parameters();

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