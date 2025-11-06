#ifndef FOCAL_FREQUENCY_LOSS_H
#define FOCAL_FREQUENCY_LOSS_H

#include <vector>
#include <optional>

#include <torch/torch.h>

// ======================================================================================
// FocalFrequencyLoss
// a frequency domain loss function for optimizing generative models.
// 
// Focal Frequency Loss for Image Reconstruction and Synthesis.In ICCV 2021.
// <https ://arxiv.org/pdf/2012.12821.pdf>
// 
// ======================================================================================

struct FocalFrequencyLossImpl : public torch::nn::Module
{
    
    explicit FocalFrequencyLossImpl(
        float loss_weight = 1.0f,
        float alpha = 1.0,
        int patch_factor = 1,
        bool ave_spectrum = false,
        bool log_matrix = false,
        bool batch_matrix = false
    );

    torch::Tensor forward(const std::vector<torch::Tensor>& pred, const torch::Tensor& target);

protected:
    float loss_weight = 1.0f;
    float alpha = 1.0;
    int patch_factor = 1;
    bool ave_spectrum = false;
    bool log_matrix = false;
    bool batch_matrix = false;

    torch::Tensor Tensor2Freq(const torch::Tensor& x) const;

    torch::Tensor CalcLoss(const torch::Tensor& weight_matrix,
        const torch::Tensor& freq_distance,
        torch::Reduction::Reduction reduction) const;

    torch::Tensor LossFormulation(const torch::Tensor& recon_freq,
        const torch::Tensor& real_freq,
        const std::optional<torch::Tensor>& matrix = std::nullopt,
        torch::Reduction::Reduction reduction = torch::Reduction::Reduction::Mean) const;

};
TORCH_MODULE(FocalFrequencyLoss);

#endif