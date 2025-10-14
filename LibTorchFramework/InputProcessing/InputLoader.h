#ifndef INPUT_LOADER_H
#define INPUT_LOADER_H

struct DataLoaderData;

#include <memory>
#include <string>
#include <vector>
#include <random>
#include <optional>

#include <torch/torch.h>

#include "./DataLoaderData.h"
#include "./InputLoadersWrapper.h"

#include "../core/Structures.h"

#include "../Settings.h"

struct InputLoaderSettings
{
    std::optional<int> subsetSize = std::nullopt;
};

class InputLoader : public std::enable_shared_from_this<InputLoader>
{
public:
    
    template <typename Dataset>
    using SimpleDataLoaderType = decltype(torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::declval<Dataset>(),
        size_t{}
    ));

    template <typename Dataset>
    using StackedDataLoaderType = decltype(
        torch::data::make_data_loader<
        torch::data::samplers::SequentialSampler>(
            std::declval<
            decltype(
                std::declval<Dataset>().map(
                    torch::data::transforms::Stack<typename Dataset::ExampleType>()
                )
                )>(),
            size_t{}
        )
    );

    //========================================================================
    
    InputLoader(RunMode type, std::weak_ptr<InputLoadersWrapper> parent);
    virtual ~InputLoader() = default;

    void SetLoaderSettings(const InputLoaderSettings& defSets);

    //==============
    // virtual method need to be overrided
    // in actuall InputLoader implementation
    //==============

    virtual size_t GetSize() const = 0;
    virtual void Load() = 0;
    virtual void FillData(size_t index, DataLoaderData& ld) = 0;

    //==============

    template <typename DatasetType>
    auto BuildDataLoader(const Settings& sets);

    template <typename DatasetType>
    auto BuildDataLoader(const Settings& sets, int& bacthesCount);

protected:
    RunMode type;
    std::weak_ptr<InputLoadersWrapper> parent;
    
    InputLoaderSettings loaderSets;

    template <typename T>
    std::vector<T> BuildSplits(const std::vector<T>& input);

    void ApplyTransform();
};

//======================================================================

template <typename T>
std::vector<T> InputLoader::BuildSplits(const std::vector<T>& input)
{
    auto parentPtr = this->parent.lock();

    size_t totalFilesCount = input.size();
    size_t offsetIndex = 0;
    size_t filesCount = 0;

    if (this->type == RunMode::TRAIN)
    {
        filesCount = static_cast<size_t>(parentPtr->GetTrainRatio() * totalFilesCount);
    }
    else if (this->type == RunMode::VALID)
    {
        filesCount = static_cast<size_t>(parentPtr->GetValRatio() * totalFilesCount);
        offsetIndex = static_cast<size_t>(parentPtr->GetTrainRatio() * totalFilesCount);
    }
    else
    {
        if (parentPtr->GetTestRatio().has_value() == false)
        {
            filesCount = totalFilesCount - static_cast<size_t>((parentPtr->GetTrainRatio() + parentPtr->GetValRatio()) * totalFilesCount);
            offsetIndex = totalFilesCount - filesCount;
        }
        else
        {
            filesCount = static_cast<size_t>(parentPtr->GetTestRatio().value() * totalFilesCount);
            offsetIndex = static_cast<size_t>(parentPtr->GetTestRatio().value() * totalFilesCount);
        }
    }

    if (filesCount == 0)
    {
        return {};
    }

    if ((loaderSets.subsetSize.has_value()) && (filesCount > loaderSets.subsetSize.value()))
    {
        filesCount = loaderSets.subsetSize.value();
    }

    // prepare indices
    std::vector<size_t> indices(totalFilesCount);
    std::iota(indices.begin(), indices.end(), 0);

    if (parentPtr->GetShuffleSeed().has_value())
    {
        std::mt19937 rng(parentPtr->GetShuffleSeed().value());
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    std::vector<T> output;
    output.reserve(filesCount);

    if (this->type == RunMode::TRAIN)
    {
        for (size_t j = 0; j < filesCount; ++j)
        {
            output.push_back(input[indices[j]]);
        }
    }
    else if (this->type == RunMode::VALID)
    {
        for (size_t j = offsetIndex; j < offsetIndex + filesCount; ++j)
        {
            output.push_back(input[indices[j]]);
        }
    }
    else
    {
        for (size_t j = totalFilesCount - filesCount; j < totalFilesCount; ++j)
        {
            output.push_back(input[indices[j]]);
        }
    }

    return output;
}

template <typename DatasetType>
auto InputLoader::BuildDataLoader(const Settings& sets)
{
    int bacthesCount;
    return this->BuildDataLoader(sets, bacthesCount);
}

template <typename DatasetType>
auto InputLoader::BuildDataLoader(const Settings& sets, int& bacthesCount)
{        
    auto ds = DatasetType(shared_from_this());
    auto dsMapped = ds.map(torch::data::transforms::Stack<typename DatasetType::ExampleType>());
    
    auto datasetSize = dsMapped.size().value();
    bacthesCount = (datasetSize + sets.batchSize - 1) / sets.batchSize;

    auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(dsMapped),        
        torch::data::DataLoaderOptions()
            .batch_size(sets.batchSize)            
            .workers(sets.numWorkers)
            .enforce_ordering(false)            
    );    

    return loader;
}

#endif