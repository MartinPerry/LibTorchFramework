#ifndef DOWNSAMPLE_2D_H
#define DOWNSAMPLE_2D_H

#include <torch/torch.h>

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

    DownSample2dImpl(int64_t in_channels,
        int64_t out_channels,
        int64_t scaleFactor,
        int64_t kernel_size = 1,
        int64_t padding = 0,
        int64_t dilation = 1,
        torch::nn::AnyModule normType = {}, // optional normalization layer
        SampleType modeType = SampleType::CLASSIC)
    {
        this->scaleFactor = scaleFactor;
        this->modeType = modeType;
        
        if (this->modeType == SampleType::CLASSIC)
        {
            torch::nn::Sequential seq;

            seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .stride(scaleFactor)
                .padding(padding)
                .dilation(dilation)
                .bias(false)));

            if (!normType.is_empty())
            {
                seq->push_back(normType);
            }
            else
            {
                seq->push_back(torch::nn::Identity());
            }

            downSample = register_module("downSample", seq);
        }
        else if (this->modeType == SampleType::SUBPIXEL)
        {
            torch::nn::Sequential seq;

            // For SUBPIXEL, input channels are multiplied by scaleFactor^2
            seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels * scaleFactor * scaleFactor, out_channels, 1)));

            if (!normType.is_empty())
            {
                seq->push_back(normType);
            }
            else
            {
                seq->push_back(torch::nn::Identity());
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