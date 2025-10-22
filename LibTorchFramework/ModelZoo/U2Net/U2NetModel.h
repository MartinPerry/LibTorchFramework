#ifndef U2NET_LIBTORCH_H
#define U2NET_LIBTORCH_H

struct ImageSize;

#include <torch/torch.h>
#include <vector>
#include <memory>

#include "../../core/AbstractModel.h"

// NOTE:
// - This is a direct translation of the provided Python U2Net model into C++ (libtorch).
// - DeformConv2d / CoordConv2d are NOT implemented.
// - This header + cpp are self-contained except for project-specific classes like AbstractTorchModel.
// If you need to inherit from AbstractTorchModel, change the base class of U2NetModel accordingly.

namespace ModelZoo
{
	namespace u2net 
	{
        enum class ConvType {
            CLASSIC = 0,
            DEFORMABLE = 1,
            COORD = 2
        };

        //===================================================================================

        struct REBNCONVImpl : public torch::nn::Module 
        {
            torch::nn::Conv2d conv{ nullptr };
            torch::nn::BatchNorm2d bn{ nullptr };
            torch::nn::ReLU relu{ nullptr };

            REBNCONVImpl(int in_ch = 3, int out_ch = 3, int dirate = 1, ConvType convType = ConvType::CLASSIC);

            torch::Tensor forward(const torch::Tensor& x);
        };
        TORCH_MODULE(REBNCONV);

        //===================================================================================

        struct RSU7Impl : public torch::nn::Module 
        {
            REBNCONV rebnconvin{ nullptr };
            REBNCONV rebnconv1{ nullptr }, rebnconv2{ nullptr }, rebnconv3{ nullptr }, rebnconv4{ nullptr }, rebnconv5{ nullptr }, rebnconv6{ nullptr }, rebnconv7{ nullptr };
            REBNCONV rebnconv6d{ nullptr }, rebnconv5d{ nullptr }, rebnconv4d{ nullptr }, rebnconv3d{ nullptr }, rebnconv2d{ nullptr }, rebnconv1d{ nullptr };
            torch::nn::MaxPool2d pool1{ nullptr }, pool2{ nullptr }, pool3{ nullptr }, pool4{ nullptr }, pool5{ nullptr };

            RSU7Impl(int in_ch = 3, int mid_ch = 12, int out_ch = 3, ConvType convType = ConvType::CLASSIC);

            torch::Tensor forward(const torch::Tensor& x);
        };
        TORCH_MODULE(RSU7);

        //===================================================================================

        struct RSU6Impl : public torch::nn::Module 
        {
            REBNCONV rebnconvin{ nullptr };
            REBNCONV rebnconv1{ nullptr }, rebnconv2{ nullptr }, rebnconv3{ nullptr }, rebnconv4{ nullptr }, rebnconv5{ nullptr }, rebnconv6{ nullptr };
            REBNCONV rebnconv5d{ nullptr }, rebnconv4d{ nullptr }, rebnconv3d{ nullptr }, rebnconv2d{ nullptr }, rebnconv1d{ nullptr };
            torch::nn::MaxPool2d pool1{ nullptr }, pool2{ nullptr }, pool3{ nullptr }, pool4{ nullptr };

            RSU6Impl(int in_ch = 3, int mid_ch = 12, int out_ch = 3, ConvType convType = ConvType::CLASSIC);
            torch::Tensor forward(const torch::Tensor& x);
        };
        TORCH_MODULE(RSU6);

        //===================================================================================

        struct RSU5Impl : public torch::nn::Module 
        {
            REBNCONV rebnconvin{ nullptr };
            REBNCONV rebnconv1{ nullptr }, rebnconv2{ nullptr }, rebnconv3{ nullptr }, rebnconv4{ nullptr }, rebnconv5{ nullptr };
            REBNCONV rebnconv4d{ nullptr }, rebnconv3d{ nullptr }, rebnconv2d{ nullptr }, rebnconv1d{ nullptr };
            torch::nn::MaxPool2d pool1{ nullptr }, pool2{ nullptr }, pool3{ nullptr };

            RSU5Impl(int in_ch = 3, int mid_ch = 12, int out_ch = 3, ConvType convType = ConvType::CLASSIC);
            torch::Tensor forward(const torch::Tensor& x);
        };
        TORCH_MODULE(RSU5);

        //===================================================================================

        struct RSU4Impl : public torch::nn::Module 
        {
            REBNCONV rebnconvin{ nullptr };
            REBNCONV rebnconv1{ nullptr }, rebnconv2{ nullptr }, rebnconv3{ nullptr }, rebnconv4{ nullptr };
            REBNCONV rebnconv3d{ nullptr }, rebnconv2d{ nullptr }, rebnconv1d{ nullptr };
            torch::nn::MaxPool2d pool1{ nullptr }, pool2{ nullptr };

            RSU4Impl(int in_ch = 3, int mid_ch = 12, int out_ch = 3, ConvType convType = ConvType::CLASSIC);
            torch::Tensor forward(const torch::Tensor& x);
        };
        TORCH_MODULE(RSU4);

        //===================================================================================
        
        struct RSU4FImpl : public torch::nn::Module 
        {
            REBNCONV rebnconvin{ nullptr };
            REBNCONV rebnconv1{ nullptr }, rebnconv2{ nullptr }, rebnconv3{ nullptr }, rebnconv4{ nullptr };
            REBNCONV rebnconv3d{ nullptr }, rebnconv2d{ nullptr }, rebnconv1d{ nullptr };

            RSU4FImpl(int in_ch = 3, int mid_ch = 12, int out_ch = 3, ConvType convType = ConvType::CLASSIC);
            torch::Tensor forward(const torch::Tensor& x);
        };
        TORCH_MODULE(RSU4F);

        //===================================================================================

        struct U2NetModel : public AbstractModel
        {            
            ConvType convType;
           
            // encoder
            RSU7 stage1{ nullptr };
            torch::nn::MaxPool2d pool12{ nullptr };
            RSU6 stage2{ nullptr };
            torch::nn::MaxPool2d pool23{ nullptr };
            RSU5 stage3{ nullptr };
            torch::nn::MaxPool2d pool34{ nullptr };
            RSU4 stage4{ nullptr };
            torch::nn::MaxPool2d pool45{ nullptr };
            RSU4F stage5{ nullptr };
            torch::nn::MaxPool2d pool56{ nullptr };
            RSU4F stage6{ nullptr };

            // decoder
            RSU4F stage5d{ nullptr };
            RSU4 stage4d{ nullptr };
            RSU5 stage3d{ nullptr };
            RSU6 stage2d{ nullptr };
            RSU7 stage1d{ nullptr };

            // side outputs
            torch::nn::Conv2d side1{ nullptr }, side2{ nullptr }, side3{ nullptr }, side4{ nullptr }, side5{ nullptr }, side6{ nullptr };
            torch::nn::Conv2d outconv{ nullptr };

            U2NetModel(int in_ch = 3, int out_ch = 1, bool small = false, ConvType convType = ConvType::CLASSIC);

            // forward returns vector of 7 tensors in same order as Python:
            // [d0, d1, d2, d3, d4, d5, d6]
            std::vector<torch::Tensor> forward(const torch::Tensor& x);

            const char* GetName() const override;

            std::vector<torch::Tensor> RunForward(DataLoaderData& batch) override;
        };
        
	}
}

#endif
