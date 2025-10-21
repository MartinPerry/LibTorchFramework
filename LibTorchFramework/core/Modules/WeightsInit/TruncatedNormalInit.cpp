#include "./TruncatedNormalInit.h"

#include <Utils/Logger.h>

double TruncatedNormalInit::mean = 0.0;
double TruncatedNormalInit::stdErr = 1.0;
double TruncatedNormalInit::a = -2.0;
double TruncatedNormalInit::b = 2.0;


double TruncatedNormalInit::norm_cdf(double x)
{
    return (1.0 + std::erf(x / std::sqrt(2.0))) / 2.0;
}

void TruncatedNormalInit::trunc_normal(torch::Tensor tensor, double mean, double std, double a, double b)
{
    if ((mean < a - 2 * std) || (mean > b + 2 * std))
    {
        MY_LOG_WARNING("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.");
    }

    double cdfA = norm_cdf((a - mean) / std);
    double cdfB = norm_cdf((b - mean) / std);

    torch::NoGradGuard no_grad;

    tensor.uniform_(2 * cdfA - 1, 2 * cdfB - 1);
    tensor.erfinv_();
    tensor.mul_(stdErr * std::sqrt(2.0));
    tensor.add_(mean);
    tensor.clamp_(a, b);
}

void TruncatedNormalInit::weights_init(torch::nn::Module& m)
{
    if (auto* conv = dynamic_cast<torch::nn::Conv2dImpl*>(&m))
    {
        trunc_normal(conv->weight, mean, stdErr, a, b);
        if (conv->bias.defined())
        {
            torch::nn::init::zeros_(conv->bias);
        }
    }
    else if (auto* linear = dynamic_cast<torch::nn::LinearImpl*>(&m))
    {
        trunc_normal(linear->weight, mean, stdErr, a, b);
        if (linear->bias.defined())
        {
            torch::nn::init::zeros_(linear->bias);
        }
    }
}

TruncatedNormalInit::TruncatedNormalInit(torch::nn::Module& model)
{
    model.apply([](torch::nn::Module& m){
        TruncatedNormalInit::weights_init(m);
    });
}