#ifndef DEFAULT_DATASET_H
#define DEFAULT_DATASET_H

class InputLoader;

#include <memory>
#include <optional>

#include <torch/torch.h>

#include "./DataLoaderData.h"

//using DataWithTarget = torch::data::Example<>;

class DefaultDataset : public torch::data::datasets::Dataset<DefaultDataset, DataLoaderData>
{
public:
    //using ExampleType = DataLoaderData;

    DefaultDataset(std::shared_ptr<InputLoader> loader);
    virtual ~DefaultDataset() = default;

    DataLoaderData get(size_t index);
    torch::optional<size_t> size() const;

protected:
    std::shared_ptr<InputLoader> loader;
};

#endif
