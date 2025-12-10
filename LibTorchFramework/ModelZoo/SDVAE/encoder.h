#ifndef SDVAE_ENCODER_H
#define SDVAE_ENCODER_H

#include <tuple>

#include <torch/torch.h>

#include "./attention.h"

namespace ModelZoo {
    namespace sdvae {

        struct VAE_EncoderImpl : torch::nn::Module
        {
            torch::nn::Sequential seq;

            VAE_EncoderImpl();

            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
                torch::Tensor x,
                torch::Tensor noise = torch::Tensor());

            torch::Tensor kl_divergence(torch::Tensor mean, torch::Tensor log_variance);
        };

        TORCH_MODULE(VAE_Encoder);
    }
}

#endif