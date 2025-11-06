#include "./SSIMLoss.h"

torch::Tensor _fspecial_gauss_1d(int size, float sigma)
{
    torch::Tensor coords = torch::arange(size, torch::kFloat);
    coords -= size / 2;
    torch::Tensor g = torch::exp(-(coords.pow(2)) / (2 * sigma * sigma));
    g /= g.sum();
    return g.unsqueeze(0).unsqueeze(0);
}

torch::Tensor gaussian_filter(const torch::Tensor& input, const torch::Tensor& win)
{
    TORCH_CHECK(win.sizes().size() >= 3, "Invalid window torch::Tensor");
    torch::Tensor out = input.clone();
    int64_t dim = input.dim();
    
    TORCH_CHECK(dim == 4 || dim == 5, "Input images should be 4D or 5D");
    
    int64_t C = input.size(1);
    for (int64_t i = 0; i < dim - 2; ++i)
    {
        if (input.size(2 + i) >= win.size(-1))
        {            
            //out = torch::conv2d(out, win.transpose(2 + i, -1), {}, 1, 0, 1, C);            
            out = torch::conv2d(
                out,
                win.transpose(2 + i, -1),
                {},
                torch::IntArrayRef({ 1, 1 }),   // stride
                torch::IntArrayRef({ 0, 0 }),   // padding
                torch::IntArrayRef({ 1, 1 }),   // dilation
                C                             // groups
            );
        }
        else
        {
            std::cerr << "Warning: Skipping Gaussian smoothing for dimension " << (2 + i) << std::endl;
        }
    }

    return out;
}

std::tuple<torch::Tensor, torch::Tensor> _ssim(const torch::Tensor& X, const torch::Tensor& Y, 
    float data_range, const torch::Tensor& win, std::tuple<float, float> K)
{
    auto [K1, K2] = K;
    float compensation = 1.0f;
    float C1 = std::pow(K1 * data_range, 2);
    float C2 = std::pow(K2 * data_range, 2);

    torch::Tensor win_d = win.to(X.device(), X.dtype());
    torch::Tensor mu1 = gaussian_filter(X, win_d);
    torch::Tensor mu2 = gaussian_filter(Y, win_d);

    torch::Tensor mu1_sq = mu1.pow(2);
    torch::Tensor mu2_sq = mu2.pow(2);
    torch::Tensor mu1_mu2 = mu1 * mu2;

    torch::Tensor sigma1_sq = compensation * (gaussian_filter(X * X, win_d) - mu1_sq);
    torch::Tensor sigma2_sq = compensation * (gaussian_filter(Y * Y, win_d) - mu2_sq);
    torch::Tensor sigma12 = compensation * (gaussian_filter(X * Y, win_d) - mu1_mu2);

    torch::Tensor cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2);
    torch::Tensor ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map;

    torch::Tensor ssim_per_channel = ssim_map.flatten(2).mean(-1);
    torch::Tensor cs = cs_map.flatten(2).mean(-1);

    return { ssim_per_channel, cs };
}


// ======================================================================================


MSSSIMLossImpl::MSSSIMLossImpl(float data_range, int win_size, float win_sigma, int channel, int spatial_dims,
    std::vector<float> weights, std::tuple<float, float> K) :
    win_size(win_size),
    data_range(data_range),
    win_sigma(win_sigma),
    weights(weights),
    K(K)   
{    
    this->win = _fspecial_gauss_1d(win_size, win_sigma).repeat({ channel, 1, 1 });
}

torch::Tensor MSSSIMLossImpl::ms_ssim(const torch::Tensor& X, const torch::Tensor& Y, 
    torch::Reduction::Reduction reduction) const
{
    TORCH_CHECK(X.sizes() == Y.sizes(), "Input images must have same shape");
    torch::Tensor X_ = X;
    torch::Tensor Y_ = Y;

    for (int d = X_.dim() - 1; d > 1; --d)
    {
        X_ = X_.squeeze(d);
        Y_ = Y_.squeeze(d);
    }

    TORCH_CHECK(X_.dim() == 4 || X_.dim() == 5, "Input images must be 4D or 5D");
    
    torch::Tensor weights_tensor = torch::tensor(weights, X_.options());
        
    int levels = weights_tensor.size(0);

    std::vector<torch::Tensor> mcs;
    torch::Tensor ssim_per_channel, cs;

    for (int i = 0; i < levels; ++i)
    {
        std::tie(ssim_per_channel, cs) = _ssim(X_, Y_, data_range, win, K);
        if (i < levels - 1)
        {
            mcs.push_back(torch::relu(cs));
            X_ = torch::avg_pool2d(X_, 2, 2);
            Y_ = torch::avg_pool2d(Y_, 2, 2);
        }
    }

    ssim_per_channel = torch::relu(ssim_per_channel);
    mcs.push_back(ssim_per_channel);

    torch::Tensor mcs_and_ssim = torch::stack(mcs, 0);
    torch::Tensor weights_view = weights_tensor.view({ -1, 1, 1 });
    torch::Tensor ms_ssim_val = torch::prod(torch::pow(mcs_and_ssim, weights_view), 0);

    if (reduction == torch::Reduction::Reduction::Mean)
    {
        return ms_ssim_val.mean();
    }
    if (reduction == torch::Reduction::Reduction::Sum)
    {
        return ms_ssim_val.sum();
    }
    return ms_ssim_val;
}

torch::Tensor MSSSIMLossImpl::forward(const torch::Tensor& X, const torch::Tensor& Y, 
    torch::Reduction::Reduction reduction)
{
    return 1.0 - ms_ssim(X, Y, reduction);
}

// ======================================================================================


SSIMLossImpl::SSIMLossImpl(float data_range, int win_size, float win_sigma, int channel, int spatial_dims,
    std::tuple<float, float> K, bool nonnegative_ssim) : 
    win_size(win_size),
    data_range(data_range),    
    win_sigma(win_sigma),
    K(K),
    nonnegative_ssim(nonnegative_ssim)
{        
    this->win = _fspecial_gauss_1d(win_size, win_sigma).repeat({ channel, 1, 1 });
}

torch::Tensor SSIMLossImpl::ssim(const torch::Tensor& X, const torch::Tensor& Y, 
    torch::Reduction::Reduction reduction) const
{
    TORCH_CHECK(X.sizes() == Y.sizes(), "Input images must have same shape");

    torch::Tensor X_ = X;
    torch::Tensor Y_ = Y;
    for (int d = X_.dim() - 1; d > 1; --d)
    {
        X_ = X_.squeeze(d);
        Y_ = Y_.squeeze(d);
    }

    TORCH_CHECK(X_.dim() == 4 || X_.dim() == 5, "Input images should be 4D or 5D");
    
    auto [ssim_per_channel, cs] = _ssim(X_, Y_, data_range, win, K);

    if (nonnegative_ssim)
    {
        ssim_per_channel = torch::relu(ssim_per_channel);
    }

    if (reduction == torch::Reduction::Reduction::Mean)
    {
        return ssim_per_channel.mean();
    }
    if (reduction == torch::Reduction::Reduction::Sum)
    {
        return ssim_per_channel.sum();
    }
    return ssim_per_channel;
}


torch::Tensor SSIMLossImpl::forward(const torch::Tensor& X, const torch::Tensor& Y, 
    torch::Reduction::Reduction reduction)
{
    return 1.0 - ssim(X, Y, reduction);
}
