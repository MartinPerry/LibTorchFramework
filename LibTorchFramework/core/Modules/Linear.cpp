#include "./Linear.h"

CustomLinearImpl::CustomLinearImpl(const torch::nn::LinearOptions& opt) : 
    CustomLinearImpl(CustomLinearOptions(opt.in_features(), opt.out_features()).bias(opt.bias()))
{    
}

CustomLinearImpl::CustomLinearImpl(const CustomLinearOptions& opt) :
    options(opt)
{
    CustomLinearImpl::reset();
}

void CustomLinearImpl::reset()
{
    weight = register_parameter("weight", torch::empty({ options.out_features(), options.in_features() }));
    
    if (options.bias()) 
    {
        bias = register_parameter("bias", torch::empty(options.out_features()));
    }
    else 
    {
        bias = register_parameter("bias", {}, /*requires_grad=*/false);
    }

    if (options.init_params())
    {
        reset_parameters();
    }
}

void CustomLinearImpl::reset_parameters()
{
    torch::nn::init::kaiming_uniform_(weight, std::sqrt(5)); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
    
    if (bias.defined()) 
    {
        auto [fan_in, fan_out] = torch::nn::init::_calculate_fan_in_and_fan_out(weight);
        const auto bound = 1 / std::sqrt(fan_in);
        torch::nn::init::uniform_(bias, -bound, bound);
    }
}

void CustomLinearImpl::pretty_print(std::ostream& stream) const
{
    stream << std::boolalpha
        << "torch::nn::Linear(in_features=" << options.in_features()
        << ", out_features=" << options.out_features()
        << ", bias=" << options.bias() << ')';
}

torch::Tensor CustomLinearImpl::forward(const torch::Tensor& input)
{    
    return torch::nn::functional::linear(input, weight, bias);
}
