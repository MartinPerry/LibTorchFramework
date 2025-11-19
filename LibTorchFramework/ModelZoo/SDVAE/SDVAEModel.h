#ifndef SDVAE_MODEL_H
#define SDVAE_MODEL_H

#include <memory>
#include <vector>
#include <optional>

#include <torch/torch.h>

#include "../../core/Modules/ModulesOptions.h"

#include "../../core/AbstractModel.h"

#include "./encoder.h"
#include "./decoder.h"

namespace ModelZoo
{
    namespace sdvae
    {

        struct SDVAEModel : public AbstractModel
        {
            VAE_Encoder encoder;
            VAE_Decoder decoder;

            SDVAEModel();
            
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
                torch::Tensor x,
                torch::Tensor noise = torch::Tensor());

            const char* GetName() const override;

            std::vector<torch::Tensor> RunForward(DataLoaderData& batch) override;
        };
    }
}

#endif