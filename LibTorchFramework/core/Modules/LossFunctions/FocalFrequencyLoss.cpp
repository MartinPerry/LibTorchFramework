#include "./FocalFrequencyLoss.h"

FocalFrequencyLossImpl::FocalFrequencyLossImpl(float loss_weight, float alpha, 
    int patch_factor, bool ave_spectrum, bool log_matrix, bool batch_matrix) : 
    loss_weight(loss_weight),
    alpha(alpha),
    patch_factor(patch_factor),
    ave_spectrum(ave_spectrum),
    log_matrix(false),
    batch_matrix(batch_matrix)
{
}

torch::Tensor FocalFrequencyLossImpl::Tensor2Freq(const torch::Tensor& x) const
{
    // Crop image patches
    int pf = patch_factor;
    auto sizes = x.sizes();
    int h = sizes[2];
    int w = sizes[3];

    TORCH_CHECK(h % pf == 0 && w % pf == 0,
        "Patch factor should be divisible by image height and width");

    std::vector<torch::Tensor> patch_list;
    int patch_h = h / pf;
    int patch_w = w / pf;

    for (int i = 0; i < pf; i++)
    {
        for (int j = 0; j < pf; j++)
        {
            patch_list.push_back(
                x.index({ torch::indexing::Slice(),
                         torch::indexing::Slice(),
                         torch::indexing::Slice(i * patch_h, (i + 1) * patch_h),
                         torch::indexing::Slice(j * patch_w, (j + 1) * patch_w) }));
        }
    }

    // Stack to patch tensor
    torch::Tensor y = torch::stack(patch_list, 1);

    // Perform 2D DFT (real-to-complex, orthonormalization)
    torch::Tensor freq = torch::fft::fft2(y, std::nullopt, {-2, -1}, "ortho");

    // Extract real and imaginary components manually
    torch::Tensor freq_real = torch::real(freq);
    torch::Tensor freq_imag = torch::imag(freq);

    // Stack them as the last dimension
    torch::Tensor freq_stacked = torch::stack({ freq_real, freq_imag }, -1);

    return freq;
}

//=====================================================================

torch::Tensor FocalFrequencyLossImpl::CalcLoss(const torch::Tensor& weight_matrix,
    const torch::Tensor& freq_distance,
    torch::Reduction::Reduction reduction) const
{
    torch::Tensor loss = weight_matrix * freq_distance;

    if (reduction == torch::Reduction::Reduction::Mean)
    {
        return loss.mean();
    }
    else if (reduction == torch::Reduction::Reduction::Sum)
    {
        return loss.sum();
    }
    
    return loss;
}

//=====================================================================

torch::Tensor FocalFrequencyLossImpl::LossFormulation(const torch::Tensor& recon_freq,
    const torch::Tensor& real_freq,
    const std::optional<torch::Tensor>& matrix,
    torch::Reduction::Reduction reduction) const
{
    torch::Tensor weight_matrix;

    if (matrix.has_value())
    {
        weight_matrix = matrix.value().detach();
    }
    else
    {
        torch::Tensor matrix_tmp = torch::pow(recon_freq - real_freq, 2);
        matrix_tmp = torch::sqrt(matrix_tmp.index({ "...", 0 }) + matrix_tmp.index({ "...", 1 }));
        matrix_tmp = torch::pow(matrix_tmp, alpha);

        if (log_matrix)
        {
            matrix_tmp = torch::log(matrix_tmp + 1.0);
        }

        if (batch_matrix)
        {
            matrix_tmp = matrix_tmp / matrix_tmp.max();
        }
        else
        {
            // Equivalent to: matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]
            auto max1 = std::get<0>(matrix_tmp.max(-1));
            auto max2 = std::get<0>(max1.max(-1));
            matrix_tmp = matrix_tmp / max2.unsqueeze(-1).unsqueeze(-1);
        }

        matrix_tmp = torch::nan_to_num(matrix_tmp, 0.0);
        matrix_tmp = torch::clamp(matrix_tmp, 0.0, 1.0);
        weight_matrix = matrix_tmp.clone().detach();
    }

    TORCH_CHECK(weight_matrix.min().item<float>() >= 0.0f &&
        weight_matrix.max().item<float>() <= 1.0f,
        "Spectrum weight matrix values must be in [0, 1]");

    torch::Tensor tmp = torch::pow(recon_freq - real_freq, 2);
    torch::Tensor freq_distance = tmp.index({ "...", 0 }) + tmp.index({ "...", 1 });

    return this->CalcLoss(weight_matrix, freq_distance, reduction);
}

//=====================================================================

torch::Tensor FocalFrequencyLossImpl::forward(const std::vector<torch::Tensor>& preds, const torch::Tensor& target)
{
    // We assume preds[0] = prediction tensor, preds may support multiple inputs
    torch::Tensor pred = preds[0];

    torch::Tensor pred_freq = this->Tensor2Freq(pred);
    torch::Tensor target_freq = this->Tensor2Freq(target);

    if (ave_spectrum)
    {
        pred_freq = torch::mean(pred_freq, 0, true);
        target_freq = torch::mean(target_freq, 0, true);
    }

    torch::Tensor loss = this->LossFormulation(pred_freq, target_freq,
        std::nullopt, torch::Reduction::Reduction::Mean);

    return loss * loss_weight;
}