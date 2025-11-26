#include "./SDVAEModel.h"

#include <Utils/Logger.h>

#include "../../InputProcessing/DataLoaderData.h"


using namespace ModelZoo::sdvae;


SDVAEModel::SDVAEModel()
{
    encoder = VAE_Encoder();
    decoder = VAE_Decoder();

    register_module("encoder", encoder);
    register_module("decoder", decoder);
}

const char* SDVAEModel::GetName() const
{
    return "SDVAEModel";
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SDVAEModel::forward(
    torch::Tensor x,
    torch::Tensor noise)
{
    
    // If encoder first conv has 1 input channel -> reduce image to 1 channel
    const auto conv0 = encoder->seq->ptr(0)->as<torch::nn::Conv2d>();
    
    if (conv0 != nullptr)
    {
        if (conv0->options.in_channels() == 1)
        {
            x = x.index({ "...", torch::indexing::Slice(0, 1), torch::indexing::Ellipsis });
        }
    }

    auto encoded = encoder->forward(x, noise);

    torch::Tensor latent = std::get<0>(encoded);
    torch::Tensor mean = std::get<1>(encoded);
    torch::Tensor logVar = std::get<2>(encoded);

    torch::Tensor recon = decoder->forward(latent);

    // If reconstruction is 1-channel -> repeat to 3 channels
    if (recon.size(1) == 1)
    {
        recon = recon.repeat({ 1, 3, 1, 1 });
    }

    return { recon, mean, logVar };
}

std::vector<torch::Tensor> SDVAEModel::RunForward(DataLoaderData& batch)
{
    //input size must be w >= 256 and h >= 256

    auto x = this->forward(batch.input);
    
    return { std::get<0>(x), batch.target };
}