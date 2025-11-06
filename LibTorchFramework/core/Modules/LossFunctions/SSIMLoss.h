#ifndef SSIM_LOSS_H
#define SSIM_LOSS_H

#include <vector>
#include <tuple>
#include <optional>

#include <torch/torch.h>

// ======================================================================================
// SSIMLoss
// ======================================================================================


struct MSSSIMLossImpl : public torch::nn::Module
{

    explicit MSSSIMLossImpl(
        float data_range = 255.0f, 
        int win_size = 11, 
        float win_sigma = 1.5f, 
        int channel = 3, 
        int spatial_dims = 2,
        std::vector<float> weights = { 0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f },
        std::tuple<float, float> K = { 0.01f, 0.03f }
    );

    torch::Tensor forward(const torch::Tensor& X, const torch::Tensor& Y, 
        torch::Reduction::Reduction reduction = torch::Reduction::Reduction::Mean);

protected:
    int win_size;
    torch::Tensor win;
    float data_range;
    float win_sigma;
    std::vector<float> weights;
    std::tuple<float, float> K;

    torch::Tensor ms_ssim(const torch::Tensor& X, const torch::Tensor& Y, 
        torch::Reduction::Reduction reduction) const;


};
TORCH_MODULE(MSSSIMLoss);

// ======================================================================================

struct SSIMLossImpl : public torch::nn::Module
{

    explicit SSIMLossImpl(
        float data_range = 255.0f, 
        int win_size = 11, 
        float win_sigma = 1.5f, 
        int channel = 3, 
        int spatial_dims = 2,
        std::tuple<float, float> K = { 0.01f, 0.03f }, 
        bool nonnegative_ssim = false
     );

    torch::Tensor forward(const torch::Tensor& X, const torch::Tensor& Y, 
        torch::Reduction::Reduction reduction = torch::Reduction::Reduction::Mean);

protected:
    int win_size;
    torch::Tensor win;
    float data_range;
    float win_sigma;
    std::tuple<float, float> K;
    bool nonnegative_ssim;

    torch::Tensor ssim(const torch::Tensor& X, const torch::Tensor& Y, 
        torch::Reduction::Reduction reduction) const;

};
TORCH_MODULE(SSIMLoss);

#endif