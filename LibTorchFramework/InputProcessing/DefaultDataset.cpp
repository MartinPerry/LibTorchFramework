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
    /*
    std::string path = options.datasetPath + data[index].first;
    auto mat = cv::imread(path);
    assert(!mat.empty());

    cv::resize(mat, mat, cv::Size(options.image_size, options.image_size));
    std::vector<cv::Mat> channels(3);
    cv::split(mat, channels);

    auto R = torch::from_blob(
        channels[2].ptr(),
        { options.image_size, options.image_size },
        torch::kUInt8);
    auto G = torch::from_blob(
        channels[1].ptr(),
        { options.image_size, options.image_size },
        torch::kUInt8);
    auto B = torch::from_blob(
        channels[0].ptr(),
        { options.image_size, options.image_size },
        torch::kUInt8);

    auto tdata = torch::cat({ R, G, B })
        .view({ 3, options.image_size, options.image_size })
        .to(torch::kFloat);
    auto tlabel = torch::tensor(data[index].second, torch::kLong);
    return { tdata, tlabel };
    */

    DataLoaderData ld(index);
    this->loader->FillData(index, ld);


    return ld;
}

torch::optional<size_t> DefaultDataset::size() const
{
    return this->loader->GetSize();
}