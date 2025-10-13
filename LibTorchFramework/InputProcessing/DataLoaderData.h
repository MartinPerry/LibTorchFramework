#ifndef DATA_LOADER_DATA_H
#define DATA_LOADER_DATA_H

#include <unordered_map>
#include <string>

#include <torch/torch.h>

#include "../Settings.h"
#include "../PerformanceSettings.h"

//=================================================================================

struct DataLoaderData
{    
    at::Tensor input;
    at::Tensor target;

    std::unordered_map<std::string, at::Tensor> additionalData;

    DataLoaderData(int64_t index) :
        index({index})
    {
    }

    DataLoaderData(std::vector<int64_t> index) :
        index(index)
    {
    }
    
    void setupDevice(const Settings& sets)
    {
        input = input.to(sets.device, input.dtype(), sets.perf.useNonBlockingTransfers);
        target = target.to(sets.device, target.dtype(), sets.perf.useNonBlockingTransfers);
        
        for (auto& [k, v] : additionalData)
        {
            v = v.to(sets.device, v.dtype(), sets.perf.useNonBlockingTransfers);
        }
    }

    int64_t GetDataIndex(size_t batchIndex = 0) const
    {
        return index[batchIndex];
    }

    const std::vector<int64_t>& GetDataIndices() const
    {
        return index;
    }

    size_t GetBatchSize() const 
    {
        return index.size();
    }

    friend struct torch::data::transforms::Stack<DataLoaderData>;

protected:
    std::vector<int64_t> index; //index within InputLoader wrapped inside vector for batching support

};

//=================================================================================
/*
With this, we can use Dataset.map with
torch::data::transforms::Stack<typename DatasetType::ExampleType>()
which will stack batches together to one tensor (B, ...)
*/

namespace torch::data::transforms
{
    template <>
    struct Stack<DataLoaderData> : public Collation<DataLoaderData>
    {
        DataLoaderData apply_batch(std::vector<DataLoaderData> ds) override
        {
            std::vector<int64_t> idx;
            idx.reserve(ds.size());

            std::vector<torch::Tensor> inputs, targets;
            inputs.reserve(ds.size());
            targets.reserve(ds.size());            

            std::unordered_map<std::string, std::vector<torch::Tensor>> additionals;

            for (auto& d : ds) 
            {
                inputs.push_back(std::move(d.input));
                targets.push_back(std::move(d.target));
                idx.push_back(d.index[0]);

                for (const auto& [k, v] : d.additionalData)
                {
                    auto it = additionals.try_emplace(k);
                    it.first->second.push_back(std::move(v));
                }
            }

            DataLoaderData d(idx);
            d.input = torch::stack(inputs);
            d.target = torch::stack(targets);
            
            for (const auto& [k, v] : additionals)
            {
                d.additionalData.try_emplace(k, torch::stack(v));
            }
            
            return std::move(d);
        }
    };
}

#endif
