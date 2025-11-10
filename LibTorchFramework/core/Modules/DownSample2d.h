#ifndef DOWNSAMPLE_2D_H
#define DOWNSAMPLE_2D_H

#include <torch/torch.h>

#include "./ModulesOptions.h"

struct DownSample2dImpl : public torch::nn::Module
{
    enum class SampleType: int
    {
        SUBPIXEL = 1,
        CLASSIC = 2
    };

    // modules
    torch::nn::Sequential downSample{ nullptr };

    // settings
    int64_t scaleFactor;
    SampleType modeType;    
    
    DownSample2dImpl(const ResampleOptions& opts,
        torch::nn::BatchNorm2d normType = { nullptr }, // optional normalization layer
        SampleType modeType = SampleType::CLASSIC) : 
        scaleFactor(opts.scaleFactor()),
        modeType(modeType)
    {        
        if (this->modeType == SampleType::CLASSIC)
        {
            torch::nn::Sequential seq;

            seq->push_back(torch::nn::Conv2d(
                torch::nn::Conv2dOptions(opts.inChannels(), opts.outChannels(), opts.kernelSize())
                .stride(scaleFactor)
                .padding(opts.padding())
                .dilation(opts.dilation())
                .bias(false)));

            if (!normType.is_empty())
            {
                seq->push_back(normType);
            }            

            downSample = register_module("downSample", seq);
        }
        else if (this->modeType == SampleType::SUBPIXEL)
        {
            torch::nn::Sequential seq;

            // For SUBPIXEL, input channels are multiplied by scaleFactor^2
            seq->push_back(torch::nn::Conv2d(
                torch::nn::Conv2dOptions(opts.inChannels() * opts.scaleFactor() * opts.scaleFactor(), opts.outChannels(), 1))
            );

            if (!normType.is_empty())
            {
                seq->push_back(normType);
            }            

            downSample = register_module("downSample", seq);
        }
    }

    torch::Tensor forward(torch::Tensor x)
    {
        if (modeType == SampleType::SUBPIXEL)
        {
            int64_t batch = x.size(0);
            int64_t channels = x.size(1);
            int64_t height = x.size(2);
            int64_t width = x.size(3);

            int64_t new_h = height / scaleFactor;
            int64_t new_w = width / scaleFactor;

            x = x.view({ batch, channels, new_h, scaleFactor, new_w, scaleFactor });
            x = x.permute({ 0, 1, 3, 5, 2, 4 });
            x = x.reshape({ batch, channels * scaleFactor * scaleFactor, new_h, new_w });
        }

        return downSample->forward(x);
    }    
};

TORCH_MODULE(DownSample2d);




#endif