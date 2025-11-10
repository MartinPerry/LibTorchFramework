#include "./ResNetModel.h"

#include "../../core/Modules/ResNetBlock.h"
#include "../../core/Modules/DownSample2d.h"

ResNetModelImpl::ResNetModelImpl(std::string type,
    std::vector<int64_t> inputShape,
    std::optional<int64_t> num_classes,    
    std::vector<int64_t> planes,
    std::vector<int64_t> layers,
    int64_t out_expansion)
{        
    this->out_expansion = out_expansion;
 
    preLayer = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(inputShape[0], planes[0], 7).stride(2).padding(3).bias(false)),
        torch::nn::BatchNorm2d(planes[0]),
        torch::nn::ReLU());
    register_module("preLayer", preLayer);

    lastInChannels = planes[0];
    maxpool = register_module("maxpool", torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions(3).stride(2).padding(1).dilation(1)
    ));
    
    int stride = 1;
    for (size_t i = 0; i < planes.size(); ++i)
    {
        auto layer = _make_layer(planes[i], layers[i], stride);
        layersList->push_back(layer);
        stride = 2;
    }
    register_module("layersList", layersList);

    createDefaultFcLayer(inputShape, num_classes);
}

torch::nn::Sequential ResNetModelImpl::_make_layer(    
    int64_t inChannels,
    int64_t blocks,
    int64_t stride)
{
    using BlockDownSample = ResNetBlock<torch::nn::ReLU, torch::nn::BatchNorm2d, DownSample2d>;
    using BlockNoReSample = ResNetBlock<>;

    torch::nn::Sequential seq;
    
    if ((stride != 1) || (lastInChannels != inChannels * out_expansion))
    {        
        seq->push_back(BlockDownSample(lastInChannels, inChannels, stride, 1, out_expansion));
    }
    else
    {
        seq->push_back(BlockNoReSample(lastInChannels, inChannels, stride, 1, out_expansion));
    }
    
    int64_t outChannels = inChannels;
    inChannels = inChannels * out_expansion;
    
    for (int i = 1; i < blocks; ++i)
    {
        seq->push_back(BlockNoReSample(inChannels, outChannels, 1, 1, out_expansion));
    }

    lastInChannels = inChannels;
    return seq;
}

torch::Tensor ResNetModelImpl::_runLayers(torch::Tensor x)
{
    x = preLayer->forward(x);
    x = maxpool->forward(x);
    

    for (auto it = layersList->begin(); it < layersList->end(); it++)
    {
        x = it->get()->as<torch::nn::Sequential>()->forward(x);        
    }
    return x;
}

void ResNetModelImpl::createDefaultFcLayer(std::vector<int64_t> inputShape, std::optional<int64_t> num_classes)
{
    auto avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({ 1, 1 }));
    auto x = torch::randn(inputShape);
    x = x.unsqueeze(0);
    x = _runLayers(x);
    x = avgpool->forward(x);
    x = x.flatten(1);

    int64_t out_features = x.size(1);
    if (!num_classes.has_value())
    {
        num_classes = out_features;
    }

    fc = torch::nn::Sequential(
        avgpool,
        torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1)),
        torch::nn::Linear(out_features, num_classes.value())
    );
    register_module("fc", fc);
}

torch::Tensor ResNetModelImpl::forward(torch::Tensor x)
{
    x = _runLayers(x);
    x = fc->forward(x);
    return x;
}