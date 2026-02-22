#include "./LoRALinear.h"

inline bool isLinearModule(const torch::nn::Module& m)
{
    return dynamic_cast<const torch::nn::LinearImpl*>(&m) != nullptr;
}

void LoRAWrap(std::shared_ptr<torch::nn::Module> m,
    const std::string& name,
    int64_t r,
    double alpha,
    double dropout,
    const std::unordered_set<std::string>& targets
)
{
    LoRAWrap(*m.get(), name, r, alpha, dropout, targets);
}


void LoRAWrap(torch::nn::Module& m,
    const std::string& name,
    int64_t r,
    double alpha,
    double dropout,
    const std::unordered_set<std::string>& targets) 
{    
    for (const auto& it : m.named_children())
    {        
        torch::nn::Module* child_ptr = it.value().get();

        if (child_ptr == nullptr)
        {
            continue;
        }
        
        torch::nn::Module& child = *child_ptr;
        
        const std::string& child_name = it.key();

        const std::string full = name.empty() ? child_name : (name + "." + child_name);

        if (targets.find(child_name) != targets.end() && isLinearModule(child)) 
        {
            // Recreate a Linear module with the same hyperparams & copy parameters,
            // then wrap it in LoRALinear and replace the child.
            auto* lin = dynamic_cast<torch::nn::LinearImpl*>(child_ptr);

            const auto in_features = lin->weight.size(1);
            const auto out_features = lin->weight.size(0);
            const bool with_bias = lin->bias.defined();

            torch::nn::LinearOptions opts(in_features, out_features);
            opts.bias(with_bias);

            // Create a new Linear holder so LoRALinear can own/register it cleanly.
            torch::nn::Linear base_new(opts);
            {
                torch::NoGradGuard ng;
                base_new->weight.copy_(lin->weight);
                if (with_bias)
                {
                    base_new->bias.copy_(lin->bias);
                }
            }

            // Replace child module by LoRALinear
            auto wrapped = LoRALinear(base_new, r, alpha, dropout);
            m.register_module(child_name, wrapped);
        }
        else 
        {
            LoRAWrap(child, full, r, alpha, dropout, targets);
        }
    }
}