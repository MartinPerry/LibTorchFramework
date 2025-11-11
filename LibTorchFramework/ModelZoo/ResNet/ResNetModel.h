#ifndef RESNET_MODEL_H
#define RESNET_MODEL_H

#include <memory>
#include <vector>
#include <optional>

#include <torch/torch.h>

#include "../../core/Modules/ModulesOptions.h"

#include "../../core/AbstractModel.h"

namespace ModelZoo
{
    namespace resnet
    {

        template <typename Activation, typename Normalization, typename ResampleType>
        class BasicResidualBlockImpl : public torch::nn::Module
        {
        public:
            BasicResidualBlockImpl(const ResidualBlockOptions& opts);

            torch::Tensor forward(torch::Tensor x);

        private:
            torch::nn::Sequential conv1{ nullptr };
            torch::nn::Sequential conv2{ nullptr };
            torch::nn::Sequential resample{ nullptr };
            torch::nn::AnyModule actFn;
        };

        //========================================================================

        class ResNetModel : public AbstractModel
        {
        public:
            ResNetModel(int64_t inChannels, int64_t w, int64_t h,
                std::optional<int64_t> num_classes = std::nullopt,
                std::vector<int64_t> planes = { 64,128,256,512 },
                std::vector<int64_t> layers = { 3,4,6,3 },
                int64_t out_expansion = 4
            );

            const char* GetName() const override;

            std::vector<torch::Tensor> RunForward(DataLoaderData& batch) override;

            torch::Tensor forward(torch::Tensor x);

        private:
            torch::nn::Sequential preLayer{ nullptr };
            torch::nn::MaxPool2d maxpool{ nullptr };
            torch::nn::ModuleList layersList;
            torch::nn::Sequential fc{ nullptr };

            int64_t out_expansion;
            int64_t lastInChannels;

            torch::nn::Sequential MakeLayer(
                int64_t inChannels,
                int64_t blocks,
                int64_t stride);

            torch::Tensor RunLayers(torch::Tensor x);
            void CreateDefaultFcLayer(int64_t inChannels, int64_t w, int64_t h, 
                std::optional<int64_t> num_classes);
        };

        //========================================================================
        //========================================================================
        //========================================================================


        template <typename Activation, typename Normalization, typename ResampleType>
        BasicResidualBlockImpl<Activation, Normalization, ResampleType>::BasicResidualBlockImpl(const ResidualBlockOptions& opts)
        {
            int64_t finalOutChannels = opts.outChannels() * opts.outExpansion();

            if constexpr (std::is_same<ResampleType, void>::value)
            {
                conv1 = torch::nn::Sequential(
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(opts.inChannels(), opts.outChannels(), 3).stride(opts.stride()).padding(1).bias(false)),
                    Normalization(opts.outChannels()),
                    Activation());
            }
            else
            {
                conv1 = torch::nn::Sequential(
                    ResampleType(
                        ResampleOptions(opts.outChannels(), opts.outChannels(), opts.stride()).kernelSize(3).padding(1),
                        Normalization(opts.outChannels())
                    ),
                    Activation());
            }
            register_module("conv1", conv1);

            conv2 = torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(opts.outChannels(), finalOutChannels, 3).stride(1).padding(1).bias(false)),
                Normalization(finalOutChannels)
            );

            register_module("conv2", conv2);

            if constexpr (std::is_same<ResampleType, void>::value)
            {
                resample = register_module("resample", torch::nn::Identity());
            }
            else
            {
                resample = register_module("resample",
                    ResampleType(
                        ResampleOptions(opts.inChannels(), finalOutChannels, opts.stride()),
                        Normalization(finalOutChannels)
                    )
                );
            }


            actFn = register_module("actFn", Activation());
        }

        template <typename Activation, typename Normalization, typename ResampleType>
        torch::Tensor BasicResidualBlockImpl<Activation, Normalization, ResampleType>::forward(torch::Tensor x)
        {
            auto out = conv1->forward(x);
            out = conv2->forward(out);
            auto residual = resample->forward(x);
            out += residual;
            out = actFn->forward(out);
            return out;
        }

    }
}

#endif