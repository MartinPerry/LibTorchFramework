#ifndef UNET_MODEL_H
#define UNET_MODEL_H

struct ImageSize;

#include <vector>
#include <optional>

#include <torch/torch.h>

#include "../../core/AbstractModel.h"

namespace ModelZoo
{
    namespace unet
    {

        struct SimpleUNetBlockImpl : public torch::nn::Module
        {
            torch::nn::Conv2d conv1{ nullptr };
            torch::nn::Conv2d conv2{ nullptr };
            torch::nn::ReLU relu{ nullptr };
            torch::nn::Dropout dropout{ nullptr };            
            bool use_dropout;

            SimpleUNetBlockImpl(int in_ch,
                int out_ch,
                bool use_padding = false,
                std::optional<double> dropout_rate = std::nullopt);

            torch::Tensor forward(torch::Tensor x);
        };

        TORCH_MODULE(SimpleUNetBlock);

        //===================================================================================

        struct EncoderImpl : public torch::nn::Module
        {
            torch::nn::ModuleList enc_blocks{ nullptr };
            torch::nn::MaxPool2d pool{ nullptr };

            EncoderImpl(int inputChannels,
                const std::vector<int>& chs = { 64, 128, 256, 512, 1024 },
                bool use_padding = false);

            std::vector<torch::Tensor> forward(torch::Tensor x);
        };

        TORCH_MODULE(Encoder);

        //===================================================================================

        struct DecoderImpl : public torch::nn::Module
        {
            std::vector<int> chs;
            torch::nn::ModuleList upconvs{ nullptr };
            torch::nn::ModuleList dec_blocks{ nullptr };

            DecoderImpl(const std::vector<int>& chs = { 1024, 512, 256, 128, 64 },
                bool use_padding = false);

            torch::Tensor crop(const torch::Tensor& img, const torch::Tensor& target);

            torch::Tensor forward(torch::Tensor x, const std::vector<torch::Tensor>& encoderFeatures);
        };

        TORCH_MODULE(Decoder);


        //===================================================================================

        class UNetModel : public AbstractModel
        {
        public:
            Encoder encoder{ nullptr };
            Decoder decoder{ nullptr };
            torch::nn::Conv2d head{ nullptr };

            int outW;
            int outH;

            UNetModel(const ImageSize& inputImSize,
                const ImageSize& outputImSize,                
                const std::vector<int>& enc_chs = { 64, 128, 256, 512, 1024 },
                const std::vector<int>& dec_chs = { 1024, 512, 256, 128, 64 }
            );

            torch::Tensor forward(torch::Tensor x);

            const char* GetName() const override;

            std::vector<torch::Tensor> RunForward(DataLoaderData& batch) override;

        protected:
        };

        //TORCH_MODULE(UNetModel); // creates module holder for NetImpl

    }
}

#endif