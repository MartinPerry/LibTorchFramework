#ifndef CUSTOM_LINEAR_MODULE_H
#define CUSTOM_LINEAR_MODULE_H

#include <torch/torch.h>

struct CustomLinearOptions
{
    CustomLinearOptions(int64_t in_features, int64_t out_features) : 
        in_features_(in_features), 
        out_features_(out_features) 
    {}

    /// size of each input sample
    TORCH_ARG(int64_t, in_features);

    /// size of each output sample
    TORCH_ARG(int64_t, out_features);

    /// If set to false, the layer will not learn an additive bias. Default: true
    TORCH_ARG(bool, bias) = true;

    /// If set to false, the layer will not init weights to random during creations. Default: true
    TORCH_ARG(bool, init_params) = true;
};

class CustomLinearImpl : public torch::nn::Cloneable<CustomLinearImpl>
{
public:
    /// The options used to configure this module.
    CustomLinearOptions options;

    /// The learned weight.
    torch::Tensor weight;

    /// The learned bias. If `bias` is false in the `options`, this tensor is
    /// undefined.
    torch::Tensor bias;

    CustomLinearImpl(int64_t in_features, int64_t out_features) :
        CustomLinearImpl(CustomLinearOptions(in_features, out_features))
    {}

    CustomLinearImpl(const torch::nn::LinearOptions& opt);

    explicit CustomLinearImpl(const CustomLinearOptions& opt);

    void reset() override;

    void reset_parameters();
    
    void pretty_print(std::ostream& stream) const override;

    torch::Tensor forward(const torch::Tensor& x);
};

TORCH_MODULE(CustomLinear);

#endif

