#ifndef SDVAE_DECODER_H
#define SDVAE_DECODER_H

#include <torch/torch.h>

#include "./attention.h"

namespace ModelZoo {
    namespace sdvae {

        class VAE_AttentionBlockImpl : public torch::nn::Module
        {
        public:
            explicit VAE_AttentionBlockImpl(int64_t channels);

            torch::Tensor forward(torch::Tensor x);

        private:
            torch::nn::GroupNorm groupnorm{ nullptr };            
            SelfAttention attention{ nullptr };
        };
        TORCH_MODULE(VAE_AttentionBlock);

        
        class VAE_ResidualBlockImpl : public torch::nn::Module
        {
        public:
            VAE_ResidualBlockImpl(int64_t in_channels, int64_t out_channels);

            torch::Tensor forward(torch::Tensor x);

        private:
            torch::nn::GroupNorm groupnorm_1{ nullptr };
            torch::nn::Conv2d conv_1{ nullptr };

            torch::nn::GroupNorm groupnorm_2{ nullptr };
            torch::nn::Conv2d conv_2{ nullptr };

            torch::nn::AnyModule residual_layer;
        };
        TORCH_MODULE(VAE_ResidualBlock);

        
        class VAE_DecoderImpl : public torch::nn::Module
        {
        public:
            VAE_DecoderImpl();

            torch::Tensor forward(torch::Tensor x);

        private:
            torch::nn::Sequential seq;
        };
        TORCH_MODULE(VAE_Decoder);


    }
}

#endif
