#ifndef UPSAMPLE_2D_H
#define UPSAMPLE_2D_H

#include <torch/torch.h>

struct UpSample2dImpl : public torch::nn::Module 
{
    enum class SampleType : int 
    {
        NEAREST_NEIGHBOR = 0,
        SUBPIXEL = 1,
        CONV_TRANPOSE = 2,
        BILINEAR = 3
    };

    // modules
    torch::nn::Conv2d conv{ nullptr };
    torch::nn::ConvTranspose2d conv_transpose{ nullptr };
    torch::nn::PixelShuffle pixel_shuffle{ nullptr };

    // settings
    int in_channels;
    int out_channels;
    int scaleFactor;
    SampleType modeType;

    // Constructor
    UpSample2dImpl(int in_channels_,
        int out_channels_,
        int scaleFactor_,
        int kernel_size = 1,
        int padding = 0,
        int dilation = 1,
        SampleType modeType_ = SampleType::NEAREST_NEIGHBOR)
        : in_channels(in_channels_),
        out_channels(out_channels_),
        scaleFactor(scaleFactor_),
        modeType(modeType_)
    {
        using namespace torch::nn;

        if (modeType == SampleType::SUBPIXEL) 
        {
            //this calculated more channels and with pixel shuffle combine
            //multiple channels to form a block of pixels in s single image

            // conv -> pixelshuffle(scaleFactor)
            Conv2dOptions conv_opts(in_channels, out_channels * scaleFactor * scaleFactor, kernel_size);
            conv_opts.stride(1).padding(padding).dilation(dilation);
            conv = register_module("conv", Conv2d(conv_opts));

            PixelShuffleOptions ps_opts(scaleFactor);
            pixel_shuffle = register_module("pixel_shuffle", PixelShuffle(ps_opts));
        }
        else if (modeType == SampleType::CONV_TRANPOSE) 
        {
            /*
             in this case, scale factor is not used directly
             https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html            

             out = [(in - 1) * stride] - [2 * padding] + [dilation * (kernel_size - 1)] + outPadding + 1
             input = 8x8
             expected out = 16x16 = [(8 - 1) * 2] - [2 * 1] + [1 * (4 - 1)] + 0 + 1
             (one possible combination !!!)
              kernel = 4
              stride = 2 (default: 1)
              padding = 1 (default: 0)
              dilation = 1 (default: 1)
              outPadding = 0 (default: 0)
             (for example if kernel is 3, outPadding should be 1 to get the same size)
            */

            ConvTranspose2dOptions ct_opts(in_channels, out_channels, kernel_size);
            ct_opts.stride(scaleFactor).padding(padding).dilation(dilation).output_padding(0);
            conv_transpose = register_module("conv_transpose", ConvTranspose2d(ct_opts));
        }
        else 
        {
            // NEAREST_NEIGHBOR or BILINEAR: we'll apply interpolate in forward and then conv
            Conv2dOptions conv_opts(in_channels, out_channels, kernel_size);
            conv_opts.stride(1).padding(padding).dilation(dilation);
            conv = register_module("conv", Conv2d(conv_opts));
        }
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        using namespace torch::nn::functional;

        if (modeType == SampleType::NEAREST_NEIGHBOR) 
        {
            auto scaleFactorDims = std::vector<double>{ static_cast<double>(scaleFactor), static_cast<double>(scaleFactor) };
            InterpolateFuncOptions opts = InterpolateFuncOptions().                
                scale_factor(std::move(scaleFactorDims)).
                mode(torch::kNearest);            
            x = interpolate(x, opts);
            x = conv->forward(x);
            return x;
        }
        else if (modeType == SampleType::BILINEAR) 
        {
            auto scaleFactorDims = std::vector<double>{ static_cast<double>(scaleFactor), static_cast<double>(scaleFactor) };

            // bilinear interpolation; for 2D set align_corners = false by default
            InterpolateFuncOptions opts = InterpolateFuncOptions().                
                scale_factor(std::move(scaleFactorDims)).
                mode(torch::kBilinear).
                align_corners(false);
            
            x = interpolate(x, opts);
            x = conv->forward(x);
            return x;
        }
        else if (modeType == SampleType::SUBPIXEL) 
        {
            x = conv->forward(x);             // produces out_channels * s * s channels
            x = pixel_shuffle->forward(x);    // pixel shuffle reduces channels -> out_channels
            return x;
        }
        else if (modeType == SampleType::CONV_TRANPOSE) 
        {
            x = conv_transpose->forward(x);
            return x;
        }

        // fallback (shouldn't happen)
        return x;
    }
};

// Expose module holder type: UpSample2d
TORCH_MODULE(UpSample2d);


#endif
