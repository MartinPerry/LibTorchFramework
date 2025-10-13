#include "./SnapshotSaver.h"

#include <unordered_set>

#include <FileUtils/Writing/RawFileWriter.h>
#include <Utils/Logger.h>

#include "../AbstractModel.h"

#include "./FreezeInfo.h"
#include "./PretrainedManager.h"

SnapshotSaver::SnapshotSaver(const AbstractModel* model) :
    model(model)
{
}


bool SnapshotSaver::Save(const std::variant<std::string, std::shared_ptr<PretrainedManager>>& pathVariant)
{

    std::string path;
   

    if (std::holds_alternative<std::shared_ptr<PretrainedManager>>(pathVariant))
    {
        std::shared_ptr<PretrainedManager> pm = std::get<std::shared_ptr<PretrainedManager>>(pathVariant);
        if (pm == nullptr)
        {
            MY_LOG_INFO("No trained model will be saved");
            return false;
        }
        path = pm->CreatePreTrainedWeightsPath(model);
    }
    else
    {
        path = std::get<std::string>(pathVariant);
    }

    if (path.empty())
    {
        MY_LOG_INFO("No trained model will be saved");
        return false;
    }

    MY_LOG_INFO("Saving trained model to %s", path.c_str());

    this->SaveParametersSerialized(path);

    
    return true;
}


void SnapshotSaver::SaveParametersSerialized(const std::string& path)
{
    torch::serialize::OutputArchive archive;

    // Save named parameters   
    torch::OrderedDict<std::string, at::Tensor> params = model->named_parameters();

    for (const auto& p : params)
    {
        const std::string name = p.key();        
        
        // Force safe representation: detach, move to CPU, make contiguous
        at::Tensor safe = p.value().detach().to(at::kCPU).contiguous();

        //MY_LOG_INFO("Saving parameter: %s", name.c_str());

        try 
        {
            archive.write(name, safe);
        }
        catch (const std::exception& e) 
        {
            MY_LOG_ERROR("Failed to write parameter '%s': %s", name.c_str(), e.what());
            // continue saving others
        }
    }

    
    // Save named buffers (optional)    
    /*
    //if there are no buffers, program crash on ~OrderedDict
    if (model->buffers().size() > 0)
    {
        torch::OrderedDict<std::string, at::Tensor> buffers = model->named_buffers();

        for (const auto& b : buffers)
        {
            const std::string name = b.key();
            at::Tensor safe = b.value().detach().to(at::kCPU).contiguous();

            //MY_LOG_INFO("Saving buffer: %s", name.c_str());

            try
            {
                archive.write(name, safe);
            }
            catch (const std::exception& e)
            {
                MY_LOG_ERROR("Failed to write buffer '%s': %s", name.c_str(), e.what());
            }
        }
    }
    */

    // Finally save to path
    archive.save_to(path);
}

/// <summary>
/// Based on load method from 
/// https://github.com/pytorch/pytorch/issues/36577
/// </summary>
/// <param name="ptPath"></param>
void SnapshotSaver::SaveParametersAsDict(const std::string& path)
{
    // Get model parameters
    torch::OrderedDict<std::string, at::Tensor> modelParams = model->named_parameters();

    // Create a dictionary to hold parameter name -> tensor mapping
    c10::Dict<std::string, at::Tensor> weights;
    for (auto const& w : modelParams)
    {                                        
        const std::string name = w.key();       
        const at::Tensor tensor = w.value().detach().to(at::kCPU).contiguous();
       
        MY_LOG_INFO("Saving parameter: %s", name.c_str());
        
        // Important: wrap both key and value in IValue explicitly
        weights.insert(name.c_str(), tensor);

    }

    // Serialize parameters using torch::pickle_save
    std::vector<char> f = torch::pickle_save(weights);

    // Write to file    
    RawFileWriter wf(path.c_str(), "wb");
    wf.Write(f);
    wf.Close();    
}
