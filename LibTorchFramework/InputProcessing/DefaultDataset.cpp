#include "./DefaultDataset.h"


#include "./InputLoader.h"

//custom dataset
//https://github.com/pytorch/examples/tree/main/cpp/custom-dataset

DefaultDataset::DefaultDataset(std::shared_ptr<InputLoader> loader) :
    loader(loader)
{
}


DataLoaderData DefaultDataset::get(size_t index)
{    
    DataLoaderData ld(index);
    this->loader->FillData(index, ld);

    return ld;
}

torch::optional<size_t> DefaultDataset::size() const
{
    return this->loader->GetSize();
}